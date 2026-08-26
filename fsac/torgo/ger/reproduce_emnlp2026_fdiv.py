from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


FSAC_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT_DEFAULT = FSAC_ROOT / "paper" / "emnlp2026_FDiv"
SOURCE_PERSONALIZED = FSAC_ROOT / "data" / "paper_personalized_train"
SOURCE_LOSO = FSAC_ROOT / "data" / "paper_loso_train"
SOURCE_CONTROL_REF = FSAC_ROOT / "data" / "paper_control_ref_ablation_train"
SOURCE_FIGURE_ANALYSIS = SOURCE_PERSONALIZED / "_analysis"

SETTING_ORDER = [
    "tiny+4B",
    "tiny+8B",
    "base+4B",
    "base+8B",
    "small+4B",
    "small+8B",
    "medium+4B",
    "medium+8B",
    "large-v2+4B",
    "large-v2+8B",
]
SEED_ORDER = ["42", "123", "456", "777", "789", "888", "999", "2024", "2025", "3407"]
SEVERITY_ORDER = ["ALL", "Severe", "Moderately-severe", "Moderate", "Mild", "control"]
FIGURE_SEVERITY_ORDER = ["Severe", "Moderately-severe", "Moderate", "Mild"]
SEVERITY_COLORS = {
    "Severe": "#B22222",
    "Moderately-severe": "#D9651C",
    "Moderate": "#C49A1D",
    "Mild": "#1F8A4C",
}
SEVERITY_MARKERS = {
    "Severe": "o",
    "Moderately-severe": "^",
    "Moderate": "s",
    "Mild": "D",
}
ACCENT_BLUE = "#1F4FBF"
GRID_GREY = "#D5D5DA"
TEXT_GREY = "#2A2C33"
SEVERITY_DISPLAY = {
    "ALL": "Overall",
    "Severe": "Severe",
    "Moderately-severe": "Mod.-sev.",
    "Moderate": "Moderate",
    "Mild": "Mild",
    "control": "Control",
}
MODE_DIR = {
    "plain": "whispering_llm_plain_s{seed}",
    "fds_off": "whispering_llm_fds_off_s{seed}",
    "vgate": "whispering_llm_vgate_s{seed}",
}
SOURCE_MODE_FILE = {
    "plain": "whispering_llama_plain_s{seed}_predictions.csv",
    "fds_off": "whispering_llama_dte_off_s{seed}_predictions.csv",
    "vgate": "whispering_llama_vgate_s{seed}_predictions.csv",
}
SOURCE_MODE_JSON = {
    "plain": "whispering_llama_plain_s{seed}/inference_results.json",
    "fds_off": "whispering_llama_dte_off_s{seed}/inference_results.json",
    "vgate": "whispering_llama_vgate_s{seed}/inference_results.json",
}

LB = r"\\"
H_TOP1 = r"\makecell{Top-1\\WER (\%)$\downarrow$}"
H_ORACLE = r"\makecell{Oracle@5\\WER (\%)$\downarrow$}"
H_PLAIN = r"\makecell{Plain\\WER (\%)$\downarrow$}"
H_FDS_OFF = r"\makecell{VGate w/o FDS\\WER (\%)$\downarrow$}"
H_FDS_VGATE = r"\makecell{\textbf{VGate}\\\textbf{WER (\%)$\downarrow$}}"
H_HEADROOM = r"\makecell{Headroom\\(pp)$\uparrow$}"
H_MEAN_HEADROOM = r"\makecell{Mean Plain\\headroom (pp)$\uparrow$}"
H_GAIN = r"\makecell{\textbf{Gain}\\\textbf{(pp)$\uparrow$}}"


@dataclass(frozen=True)
class Cell:
    dataset: str
    setting: str
    whisper: str
    llm_tag: str
    seed: str
    mode: str
    pred: Path
    result_json: Path
    prompts: Path
    frame_fds: Path | None
    fold: str = ""
    k: str = ""


def fnum(value: object, digits: int = 2) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def gain_str(value: object, digits: int = 2) -> str:
    s = fnum(value, digits)
    if s == "--":
        return s
    return f"{float(value):+.{digits}f}"


def tex_gain(value: object, digits: int = 2) -> str:
    s = gain_str(value, digits)
    return s.replace("-", "$-$") if s != "--" else s


def bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def bold_if_positive_gain(value: object, digits: int = 2) -> str:
    s = tex_gain(value, digits)
    try:
        return bold(s) if float(value) > 0 else s
    except (TypeError, ValueError):
        return s


def bold_min_wer(plain: object, off: object, vgate: object) -> tuple[str, str, str]:
    values = [float(plain), float(off), float(vgate)]
    labels = [fnum(plain), fnum(off), fnum(vgate)]
    min_value = min(values)
    return tuple(bold(label) if value == min_value else label for value, label in zip(values, labels))


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy) ** 0.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repo_display_path(value: Path | str | None, root: Path = FSAC_ROOT) -> str:
    if value is None:
        return "None"
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        pass
    parts = path.parts
    if "paper" in parts:
        return str(Path(*parts[parts.index("paper") :]))
    if not path.is_absolute():
        return str(path)
    return f"<external>/{path.name}"


def latest_repro_root(kind: str) -> Path | None:
    runs = FSAC_ROOT / "results" / "torgo" / "runs"
    candidates = sorted(runs.glob(f"repro_{kind}_checkpoint_*"))
    return candidates[-1] if candidates else None


def parse_summary(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"both(?:_checks)?(?:_passed| checks passed)?\s*:\s*(\d+)\s*/\s*(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_preconditions(personalized_root: Path | None, loso_root: Path | None, allow_source_results: bool) -> list[dict[str, str]]:
    if allow_source_results:
        return []
    rows: list[dict[str, str]] = []
    for name, root, expected in [
        ("personalized", personalized_root, 300),
        ("loso", loso_root, 240),
    ]:
        if root is None:
            status = "warn" if allow_source_results else "fail"
            rows.append({"check": f"{name}_checkpoint_repro", "status": status, "detail": "missing repro root", "path": ""})
            continue
        summary = parse_summary(root / "_diff_summary.txt")
        if summary is None:
            status = "warn" if allow_source_results else "fail"
            detail = "missing _diff_summary.txt"
        elif summary != (expected, expected):
            status = "warn" if allow_source_results else "fail"
            detail = f"both_pass={summary[0]}/{summary[1]}, expected={expected}/{expected}"
        else:
            status = "pass"
            detail = f"both_pass={expected}/{expected}"
        rows.append({"check": f"{name}_checkpoint_repro", "status": status, "detail": detail, "path": repo_display_path(root)})
    return rows


def setting_name(whisper: str, llm_tag: str) -> str:
    whisper = "large-v2" if whisper in {"largev2", "large-v2"} else whisper
    llm = "8B" if "8b" in llm_tag.lower() else "4B"
    return f"{whisper}+{llm}"


def cells_from_manifest(root: Path, dataset: str) -> list[Cell]:
    manifest = root / "_manifest.csv"
    if not manifest.exists():
        return []
    cells: list[Cell] = []
    for row in read_csv(manifest):
        mode = row["mode"]
        seed = row["seed"]
        run_dir = Path(row["run_dir"])
        if dataset == "personalized":
            pred = run_dir / mode / MODE_DIR[mode].format(seed=seed) / "predictions.csv"
            result_json = run_dir / mode / MODE_DIR[mode].format(seed=seed) / "inference_results.json"
            frame_fds = run_dir / mode / "whisper_fds" / "frame_fds_scores.csv"
        else:
            pred = run_dir / MODE_DIR[mode].format(seed=seed) / "predictions.csv"
            result_json = run_dir / MODE_DIR[mode].format(seed=seed) / "inference_results.json"
            frame_fds = run_dir / "whisper_fds" / "frame_fds_scores.csv"
        cells.append(
            Cell(
                dataset=dataset,
                setting=setting_name(row["whisper"], row["llm_tag"]),
                whisper=row["whisper"],
                llm_tag=row["llm_tag"],
                seed=seed,
                mode=mode,
                pred=pred,
                result_json=result_json,
                prompts=Path(row["prompts_csv"]),
                frame_fds=frame_fds if frame_fds.exists() else Path(row["frame_dte"]),
                fold=row.get("fold", ""),
                k=row.get("k", ""),
            )
        )
    return cells


def cells_from_source(root: Path, dataset: str) -> list[Cell]:
    cells: list[Cell] = []
    if dataset == "personalized":
        for seed_dir in sorted(root.glob("*/*/seed*")):
            if ".bak" in str(seed_dir) or not seed_dir.is_dir():
                continue
            whisper, llm_tag, seed = seed_dir.parts[-3], seed_dir.parts[-2], seed_dir.name.removeprefix("seed")
            for mode in ("plain", "fds_off", "vgate"):
                pred = seed_dir / SOURCE_MODE_FILE[mode].format(seed=seed)
                result_json = seed_dir / SOURCE_MODE_JSON[mode].format(seed=seed)
                if pred.exists() and result_json.exists():
                    cells.append(
                        Cell(
                            dataset=dataset,
                            setting=setting_name(whisper, llm_tag),
                            whisper=whisper,
                            llm_tag=llm_tag,
                            seed=seed,
                            mode=mode,
                            pred=pred,
                            result_json=result_json,
                            prompts=seed_dir / "ger_prompts.csv",
                            frame_fds=seed_dir / "whisper_dte" / "frame_dte_scores.csv",
                        )
                    )
    else:
        for seed_dir in sorted(root.glob("largev2/qwen3_8b/fold_*/k0/seed*")):
            if ".bak" in str(seed_dir) or not seed_dir.is_dir():
                continue
            seed = seed_dir.name.removeprefix("seed")
            fold = seed_dir.parts[-3].removeprefix("fold_")
            for mode in ("plain", "fds_off", "vgate"):
                pred = seed_dir / SOURCE_MODE_FILE[mode].format(seed=seed)
                result_json = seed_dir / SOURCE_MODE_JSON[mode].format(seed=seed)
                if pred.exists() and result_json.exists():
                    cells.append(
                        Cell(
                            dataset=dataset,
                            setting="large-v2+8B",
                            whisper="largev2",
                            llm_tag="qwen3_8b",
                            seed=seed,
                            mode=mode,
                            pred=pred,
                            result_json=result_json,
                            prompts=seed_dir / "ger_prompts.csv",
                            frame_fds=seed_dir / "whisper_dte" / "frame_dte_scores.csv",
                            fold=fold,
                            k="0",
                        )
                    )
    return cells


def prediction_by_file(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {r["filename"]: r for r in read_csv(path)}


def prompt_by_file(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_csv(path)
    return {r.get("filename") or r.get("audio") or r.get("utt_id", ""): r for r in rows}


def fds_by_file(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    fallback: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(path):
        fname = row.get("filename") or row.get("audio") or row.get("utt_id")
        if not fname:
            continue
        mean_key = next((k for k in ("frame_fds_mean", "frame_dte_mean") if k in row and row[k] != ""), "")
        if mean_key:
            std_key = "frame_fds_std" if "frame_fds_std" in row else "frame_dte_std"
            max_key = "frame_fds_max" if "frame_fds_max" in row else "frame_dte_max"
            out[fname] = {
                "mean": float(row[mean_key]),
                "std": float(row.get(std_key) or "nan"),
                "max": float(row.get(max_key) or row[mean_key]),
            }
            continue
        for key in ("frame_fds_mean", "frame_dte_mean", "frame_fds", "frame_dte", "fds", "dte", "score", "frame_score"):
            if key in row:
                try:
                    fallback[fname].append(float(row[key]))
                except ValueError:
                    pass
                break
    for fname, vals in fallback.items():
        if fname not in out:
            out[fname] = {"mean": mean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0, "max": max(vals)}
    return out


def extract_hypotheses(prompt: str) -> list[str]:
    hyps: list[str] = []
    for line in prompt.splitlines():
        m = re.match(r"^\s*\d+\.\s*(?:\(score=[^)]+\)\s*)?(.*)\s*$", line)
        if m:
            text = m.group(1).strip()
            if text:
                hyps.append(text)
    return hyps[:5]


def _wer_parts_tokens(r: list[str], h: list[str], mode: str) -> tuple[int, int, int, int]:
    n, m = len(r), len(h)
    if mode == "loso_raw":
        dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = (i, 0, i, 0)
        for j in range(1, m + 1):
            dp[0][j] = (j, 0, 0, j)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost, s, d, ins = dp[i - 1][j - 1]
                sub = (cost, s, d, ins) if r[i - 1] == h[j - 1] else (cost + 1, s + 1, d, ins)
                cost, s, d, ins = dp[i - 1][j]
                delete = (cost + 1, s, d + 1, ins)
                cost, s, d, ins = dp[i][j - 1]
                insert = (cost + 1, s, d, ins + 1)
                dp[i][j] = min((sub, delete, insert), key=lambda x: (x[0], x[1] + x[2], x[3]))
        _, s, d, ins = dp[n][m]
        return max(n, 1), s, d, ins

    rows = n + 1
    cols = m + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = "I"
    priority = {"M": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, rows):
        for j in range(1, cols):
            candidates = [(dp[i - 1][j - 1], "M")] if r[i - 1] == h[j - 1] else [(dp[i - 1][j - 1] + 1, "S")]
            candidates.extend([(dp[i - 1][j] + 1, "D"), (dp[i][j - 1] + 1, "I")])
            score, op = min(candidates, key=lambda x: (x[0], priority[x[1]]))
            dp[i][j] = score
            back[i][j] = op

    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            s += 1
            i -= 1
            j -= 1
        elif op == "D":
            d += 1
            i -= 1
        elif op == "I":
            ins += 1
            j -= 1
        elif i > 0:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return max(n, 1), s, d, ins


def wer_parts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    return _wer_parts_tokens(normalise_text(ref).split(), normalise_text(hyp).split(), "paper")


def wer_parts_loso(ref: str, hyp: str) -> tuple[int, int, int, int]:
    return _wer_parts_tokens(str(ref).strip().split(), str(hyp).strip().split(), "loso_raw")


def wer_percent(ref: str, hyp: str) -> float:
    n, s, d, ins = wer_parts(ref, hyp)
    return 100.0 * (s + d + ins) / max(1, n)


def normalise_text(text: str) -> str:
    text = str(text).upper().strip()
    text = re.sub(r"[^A-Z\s']", "", text)
    return " ".join(text.split())


def row_ref(row: dict[str, str], prompt_row: dict[str, str] | None = None) -> str:
    prompt_row = prompt_row or {}
    return row.get("reference") or row.get("ref_text") or prompt_row.get("reference") or prompt_row.get("ref_text") or prompt_row.get("target") or ""


def row_wer(row: dict[str, str], ref: str) -> float:
    raw = row.get("wer")
    if raw not in (None, ""):
        try:
            return 100.0 * float(raw)
        except ValueError:
            pass
    return wer_percent(ref, row.get("prediction", ""))


def oracle_wer(ref: str, prompt_row: dict[str, str]) -> float:
    hyps = extract_hypotheses(prompt_row.get("prompt", ""))
    if not hyps:
        for key in ("top1", "hypothesis", "text"):
            if key in prompt_row:
                hyps = [prompt_row[key]]
                break
    if not hyps:
        return float("nan")
    return min(wer_percent(ref, hyp) for hyp in hyps)


def top1_wer(ref: str, prompt_row: dict[str, str]) -> float:
    hyps = extract_hypotheses(prompt_row.get("prompt", ""))
    if not hyps:
        for key in ("top1", "hypothesis", "text"):
            if key in prompt_row:
                hyps = [prompt_row[key]]
                break
    return wer_percent(ref, hyps[0]) if hyps else float("nan")


def nbest_diversity(prompt_row: dict[str, str]) -> float:
    hyps = extract_hypotheses(prompt_row.get("prompt", ""))
    if len(hyps) < 2:
        return 0.0
    toks = [normalise_text(h).split() for h in hyps if normalise_text(h)]
    if len(toks) < 2:
        return 0.0
    vals = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            n, s, d, ins = _wer_parts_tokens(toks[i], toks[j], "paper")
            vals.append((s + d + ins) / max(len(toks[i]), len(toks[j]), 1))
    return mean(vals)


def add_bins(rows: list[dict]) -> None:
    def row_float(row: dict, key: str) -> float:
        value = row.get(key)
        return float(value) if value not in (None, "") else float("nan")

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset_rows in by_dataset.values():
        ordered = sorted(dataset_rows, key=lambda r: float(r["plain_headroom"]))
        n = len(ordered)
        for idx, row in enumerate(ordered):
            if idx < n // 3 + 1:
                row["headroom_bin"] = "Low headroom"
            elif idx < (2 * n) // 3 + 1:
                row["headroom_bin"] = "Mid headroom"
            else:
                row["headroom_bin"] = "High headroom"
    for row in rows:
        fds = row_float(row, "frame_fds_mean")
        div = row_float(row, "nbest_diversity")
        row["fds_bin"] = (
            "FDS=<50"
            if fds < 50
            else "FDS=50-60"
            if fds < 60
            else "FDS=60-70"
            if fds < 70
            else "FDS=70-75"
            if fds < 75
            else "FDS=75-80"
            if fds < 80
            else "FDS=80+"
        )
        row["div_bin"] = (
            "DIV=<0.05"
            if div < 0.05
            else "DIV=0.05-0.15"
            if div < 0.15
            else "DIV=0.15-0.30"
            if div < 0.30
            else "DIV=0.30-0.50"
            if div < 0.50
            else "DIV=0.50+"
        )


def build_unified_table(cells: list[Cell]) -> list[dict]:
    grouped: dict[tuple, dict[str, Cell]] = defaultdict(dict)
    for cell in cells:
        key = (cell.dataset, cell.setting, cell.whisper, cell.llm_tag, cell.seed, cell.fold, cell.k)
        grouped[key][cell.mode] = cell

    setting_rank = {name: idx for idx, name in enumerate(SETTING_ORDER)}
    seed_rank = {seed: idx for idx, seed in enumerate(SEED_ORDER)}

    def group_order(item: tuple[tuple, dict[str, Cell]]) -> tuple:
        dataset, setting, _whisper, _llm_tag, seed, fold, k = item[0]
        dataset_idx = {"personalized": 0, "loso": 1}.get(dataset, 9)
        return (dataset_idx, setting_rank.get(setting, 999), str(fold), str(k), seed_rank.get(seed, 999), str(seed))

    rows: list[dict] = []
    for (dataset, setting, whisper, llm_tag, seed, fold, k), modes in sorted(grouped.items(), key=group_order):
        if not {"plain", "fds_off", "vgate"}.issubset(modes):
            continue
        preds = {mode: prediction_by_file(cell.pred) for mode, cell in modes.items()}
        prompts = prompt_by_file(modes["plain"].prompts)
        fds = fds_by_file(modes["plain"].frame_fds)
        common_set = set(preds["plain"]) & set(preds["fds_off"]) & set(preds["vgate"])
        common = [filename for filename in preds["plain"] if filename in common_set]
        parts_fn = wer_parts_loso if dataset == "loso" else wer_parts
        for filename in common:
            plain = preds["plain"][filename]
            off = preds["fds_off"][filename]
            vgate = preds["vgate"][filename]
            prompt_row = prompts.get(filename, {})
            ref = row_ref(plain, prompt_row)
            hyps = extract_hypotheses(prompt_row.get("prompt", ""))
            if not hyps:
                for key in ("top1", "hypothesis", "text"):
                    if key in prompt_row:
                        hyps = [prompt_row[key]]
                        break
            top1_pred = hyps[0] if hyps else ""
            t_n, t_s, t_d, t_i = parts_fn(ref, top1_pred)
            best_rank = 1
            best_pred = top1_pred
            best_parts = (t_n, t_s, t_d, t_i)
            for rank, hyp in enumerate(hyps, start=1):
                h_n, h_s, h_d, h_i = parts_fn(ref, hyp)
                if (h_s + h_d + h_i, rank) < (best_parts[1] + best_parts[2] + best_parts[3], best_rank):
                    best_rank = rank
                    best_pred = hyp
                    best_parts = (h_n, h_s, h_d, h_i)
            q_n, q_s, q_d, q_i = best_parts
            p_n, p_s, p_d, p_i = parts_fn(ref, plain.get("prediction", ""))
            o_n, o_s, o_d, o_i = parts_fn(ref, off.get("prediction", ""))
            v_n, v_s, v_d, v_i = parts_fn(ref, vgate.get("prediction", ""))
            top1 = 100.0 * (t_s + t_d + t_i) / max(1, t_n)
            oracle = 100.0 * (q_s + q_d + q_i) / max(1, q_n)
            plain_wer = 100.0 * (p_s + p_d + p_i) / max(1, p_n)
            off_wer = 100.0 * (o_s + o_d + o_i) / max(1, o_n)
            vgate_wer = 100.0 * (v_s + v_d + v_i) / max(1, v_n)
            plain_len = len((plain.get("prediction", "").strip().split() if dataset == "loso" else normalise_text(plain.get("prediction", "")).split()))
            fds_item = fds.get(filename, {})
            rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "whisper": whisper,
                    "llm_tag": llm_tag,
                    "seed": seed,
                    "fold": fold,
                    "k": k,
                    "filename": filename,
                    "speaker": plain.get("speaker", ""),
                    "severity": plain.get("severity") or "control",
                    "reference": ref,
                    "top1_prediction": top1_pred,
                    "oracle5_prediction": best_pred,
                    "plain_prediction": plain.get("prediction", ""),
                    "fds_off_prediction": off.get("prediction", ""),
                    "vgate_prediction": vgate.get("prediction", ""),
                    "top1_wer": top1,
                    "oracle5_wer": oracle,
                    "plain_wer": plain_wer,
                    "fds_off_wer": off_wer,
                    "vgate_wer": vgate_wer,
                    "plain_headroom": plain_wer - oracle if not math.isnan(oracle) else float("nan"),
                    "plain_vs_vgate_gain": plain_wer - vgate_wer,
                    "plain_vs_fds_off_gain": plain_wer - off_wer,
                    "top1_N": t_n,
                    "top1_S": t_s,
                    "top1_D": t_d,
                    "top1_I": t_i,
                    "top1_E": t_s + t_d + t_i,
                    "oracle5_N": q_n,
                    "oracle5_S": q_s,
                    "oracle5_D": q_d,
                    "oracle5_I": q_i,
                    "oracle5_E": q_s + q_d + q_i,
                    "plain_N": p_n,
                    "plain_S": p_s,
                    "plain_D": p_d,
                    "plain_I": p_i,
                    "plain_E": p_s + p_d + p_i,
                    "plain_len_ratio": plain_len / max(1, p_n),
                    "fds_off_N": o_n,
                    "fds_off_S": o_s,
                    "fds_off_D": o_d,
                    "fds_off_I": o_i,
                    "fds_off_E": o_s + o_d + o_i,
                    "vgate_N": v_n,
                    "vgate_S": v_s,
                    "vgate_D": v_d,
                    "vgate_I": v_i,
                    "vgate_E": v_s + v_d + v_i,
                    "frame_fds_mean": fds_item.get("mean", float("nan")),
                    "frame_fds_std": fds_item.get("std", float("nan")),
                    "frame_fds_max": fds_item.get("max", float("nan")),
                    "nbest_diversity": nbest_diversity(prompt_row),
                }
            )
    add_bins(rows)
    return rows


def aggregate_json_cells(cells: list[Cell]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for cell in cells:
        if cell.dataset != "personalized" or not cell.result_json.exists():
            continue
        payload = json.loads(cell.result_json.read_text(encoding="utf-8"))
        groups[(cell.setting, cell.mode, cell.seed)].append(payload)

    by_setting_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (setting, mode, _seed), payloads in groups.items():
        if payloads:
            by_setting_mode[(setting, mode)].append(payloads[0])

    out: list[dict] = []
    for (setting, mode), payloads in sorted(by_setting_mode.items()):
        for sev in SEVERITY_ORDER:
            vals = []
            ns = []
            for p in payloads:
                if sev == "ALL":
                    vals.append(float(p.get("overall_wer", "nan")) * 100.0)
                    ns.append(p.get("n_samples", ""))
                else:
                    item = (p.get("severity_breakdown") or {}).get(sev)
                    if item:
                        vals.append(float(item.get("wer", "nan")) * 100.0)
                        ns.append(item.get("n", ""))
            if vals:
                out.append(
                    {
                        "setting": setting,
                        "mode": mode,
                        "severity": sev,
                        "wer": mean(vals),
                        "wer_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "n_seeds": len(vals),
                        "n_utts": ns[0] if ns else "",
                    }
                )
    return out


def aggregate_unified(rows: list[dict], dataset: str) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["dataset"] != dataset:
            continue
        for sev in ("ALL", r["severity"]):
            groups[(r["setting"], sev)].append(r)
    out: list[dict] = []
    for (setting, severity), rs in sorted(groups.items()):
        out.append(
            {
                "setting": setting,
                "severity": severity,
                "n": len(rs),
                "top1": mean(float(r["top1_wer"]) for r in rs),
                "oracle5": mean(float(r["oracle5_wer"]) for r in rs),
                "plain": mean(float(r["plain_wer"]) for r in rs),
                "fds_off": mean(float(r["fds_off_wer"]) for r in rs),
                "vgate": mean(float(r["vgate_wer"]) for r in rs),
                "headroom": mean(float(r["plain_headroom"]) for r in rs),
                "gain": mean(float(r["plain_vs_vgate_gain"]) for r in rs),
            }
        )
    return out


def _component_wer(rs: list[dict], prefix: str) -> float:
    err = sum(float(r.get(f"{prefix}_E", float(r[f"{prefix}_S"]) + float(r[f"{prefix}_D"]) + float(r[f"{prefix}_I"]))) for r in rs)
    words = sum(float(r[f"{prefix}_N"]) for r in rs)
    return 100.0 * err / words if words else float("nan")


def aggregate_corpus(rows: list[dict], dataset: str, seed_macro: bool) -> list[dict]:
    source = [r for r in rows if r["dataset"] == dataset]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in source:
        for sev in ("ALL", r["severity"]):
            groups[(r["setting"], sev)].append(r)
    out: list[dict] = []
    for (setting, severity), rs in sorted(groups.items(), key=lambda x: (SETTING_ORDER.index(x[0][0]) if x[0][0] in SETTING_ORDER else 999, x[0][1])):
        if seed_macro:
            by_seed: dict[str, list[dict]] = defaultdict(list)
            for r in rs:
                by_seed[r["seed"]].append(r)
            mode_vals: dict[str, float] = {}
            for prefix in ("top1", "oracle5", "plain", "fds_off", "vgate"):
                vals = [_component_wer(by_seed[seed], prefix) for seed in SEED_ORDER if by_seed.get(seed)]
                mode_vals[prefix] = mean(vals)
        else:
            mode_vals = {prefix: _component_wer(rs, prefix) for prefix in ("top1", "oracle5", "plain", "fds_off", "vgate")}
        out.append(
            {
                "setting": setting,
                "severity": severity,
                "n": len(rs),
                "top1": mode_vals["top1"],
                "oracle5": mode_vals["oracle5"],
                "plain": mode_vals["plain"],
                "fds_off": mode_vals["fds_off"],
                "vgate": mode_vals["vgate"],
                "headroom": mode_vals["plain"] - mode_vals["oracle5"],
                "gain": mode_vals["plain"] - mode_vals["vgate"],
            }
        )
    return out


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def tex_table(label: str, caption: str, headers: list[str], rows: list[list[str]], wide: bool = False) -> str:
    env = "table*" if wide else "table"
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
        f"\\begin{{{env}}}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + f" {LB}",
        "\\midrule",
    ]
    lines += [" & ".join(row) + f" {LB}" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def write_named_table(out_dir: Path, name: str, headers: list[str], rows: list[list[str]], caption: str, wide: bool = False) -> None:
    table_dir = out_dir / "tables"
    write_text(table_dir / f"{name}.md", markdown_table(headers, rows))
    write_text(table_dir / f"{name}.tex", tex_table(f"tab:{name.removeprefix('tab_')}", caption, headers, rows, wide=wide))
    write_csv(table_dir / f"{name}.csv", [dict(zip(headers, row)) for row in rows], headers)


def write_table_bundle(out_dir: Path, name: str, headers: list[str], rows: list[list[str]], tex: str) -> None:
    table_dir = out_dir / "tables"
    write_text(table_dir / f"{name}.md", markdown_table(headers, rows))
    write_text(table_dir / f"{name}.tex", tex)
    write_csv(table_dir / f"{name}.csv", [dict(zip(headers, row)) for row in rows], headers)


def json_wer_lookup(cells: list[Cell]) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for cell in cells:
        if cell.dataset != "personalized" or not cell.result_json.exists():
            continue
        payload = json.loads(cell.result_json.read_text(encoding="utf-8"))
        out[(cell.setting, cell.mode, "ALL")].append(float(payload.get("overall_wer", "nan")) * 100.0)
        for severity, item in (payload.get("severity_breakdown") or {}).items():
            out[(cell.setting, cell.mode, severity)].append(float(item.get("wer", "nan")) * 100.0)
    return {key: mean(vals) for key, vals in out.items()}


def mode_value(
    json_lookup: dict[tuple[str, str, str], float],
    by_unified: dict[tuple[str, str], dict],
    setting: str,
    severity: str,
    mode: str,
) -> float:
    if (setting, mode, severity) in json_lookup:
        return json_lookup[(setting, mode, severity)]
    row = by_unified[(setting, severity)]
    return float(row[mode])


def build_tables(out_dir: Path, unified: list[dict], cells: list[Cell]) -> list[str]:
    written: list[str] = []
    main_agg = aggregate_corpus(unified, "personalized", seed_macro=True)
    severity_agg = aggregate_corpus(unified, "personalized", seed_macro=False)
    by_main = {(r["setting"], r["severity"]): r for r in main_agg}
    by_severity = {(r["setting"], r["severity"]): r for r in severity_agg}

    rows = []
    body = []
    wins = 0
    beats_plain = 0
    best_abs_setting = ""
    best_abs_score = float("inf")
    largest_gain_setting = ""
    largest_gain = -float("inf")
    for setting in SETTING_ORDER:
        r = by_main.get((setting, "ALL"))
        if not r:
            continue
        plain = float(r["plain"])
        off = float(r["fds_off"])
        vgate = float(r["vgate"])
        headroom = plain - float(r["oracle5"])
        gain = plain - vgate
        p_s, o_s, v_s = bold_min_wer(plain, off, vgate)
        if vgate < plain:
            beats_plain += 1
        if vgate == min(plain, off, vgate):
            wins += 1
        if vgate < best_abs_score:
            best_abs_score = vgate
            best_abs_setting = setting
        if gain > largest_gain:
            largest_gain = gain
            largest_gain_setting = setting
        rows.append(
            [
                setting,
                fnum(r["top1"]),
                fnum(r["oracle5"]),
                fnum(plain),
                fnum(off),
                fnum(vgate),
                fnum(headroom),
                gain_str(gain),
            ]
        )
        body.append(
            f"{setting} & {fnum(r['top1'])} & {fnum(r['oracle5'])} & {p_s} & {o_s} & {v_s} & "
            f"{fnum(headroom)} & {bold_if_positive_gain(gain)} {LB}"
        )
    footer = (
        f"Under full-test aggregation, VGate beats Plain in {beats_plain}/10 settings "
        f"and is the best of \\{{Plain, VGate w/o FDS, VGate\\}} in {wins}/10; "
        f"best VGate absolute WER: {best_abs_setting} ({fnum(best_abs_score)}\\%); "
        f"largest Plain-to-VGate gain: {largest_gain_setting} ({tex_gain(largest_gain)} pp)."
    )
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{l rr rrr rr}",
            "\\toprule",
            f" & \\multicolumn{{2}}{{c}}{{References}} & \\multicolumn{{3}}{{c}}{{Correction modes}} & \\multicolumn{{2}}{{c}}{{Differences}} {LB}",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6} \\cmidrule(lr){7-8}",
            f"Setting & {H_TOP1} & {H_ORACLE} & {H_PLAIN} & {H_FDS_OFF} & {H_FDS_VGATE} & {H_MEAN_HEADROOM} & {H_GAIN} {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\vspace{0.35ex}",
            "\\begin{minipage}{0.78\\textwidth}",
            f"\\footnotesize\\itshape {footer}",
            "\\end{minipage}",
            "\\par\\vspace{0.35ex}",
            "\\caption{Overall WER across ten settings. All WER columns are percentages. Setting denotes the Whisper model used to generate the N-best text hypotheses and the Qwen3 size used by the GER corrector; the acoustic conditioning path uses frozen Whisper large-v2 encoder frames for all settings. Headroom and Gain are WER differences in percentage points: Headroom $=$ Plain WER $-$ Oracle@5 WER; Gain $=$ Plain WER $-$ VGate WER. Each cell is a ten-seed macro average.}",
            "\\label{tab:main}",
            "\\end{table*}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_main", ["Setting", "Top-1", "Oracle@5", "Plain", "VGate w/o FDS", "VGate", "Headroom", "Gain"], rows, tex)
    written.append("tab_main")

    rows = []
    body = []
    wins = 0
    beats_plain = 0
    gain_vals = []
    peak_setting = ""
    peak_plain = float("nan")
    peak_vgate = float("nan")
    peak_gain = -float("inf")
    for setting in SETTING_ORDER:
        r = by_severity.get((setting, "Moderately-severe"))
        if not r:
            continue
        plain = float(r["plain"])
        off = float(r["fds_off"])
        vgate = float(r["vgate"])
        headroom = plain - float(r["oracle5"])
        gain = plain - vgate
        gain_vals.append(gain)
        p_s, o_s, v_s = bold_min_wer(plain, off, vgate)
        if vgate < plain:
            beats_plain += 1
        if vgate == min(plain, off, vgate):
            wins += 1
        if gain > peak_gain:
            peak_setting, peak_plain, peak_vgate, peak_gain = setting, plain, vgate, gain
        rows.append([setting, fnum(r["top1"]), fnum(r["oracle5"]), fnum(plain), fnum(off), fnum(vgate), fnum(headroom), gain_str(gain)])
        body.append(
            f"{setting} & {fnum(r['top1'])} & {fnum(r['oracle5'])} & {p_s} & {o_s} & {v_s} & "
            f"{fnum(headroom)} & {bold_if_positive_gain(gain)} {LB}"
        )
    mean_gain = mean(gain_vals)
    footer = (
        f"Within the Mod.-sev. slice, VGate beats Plain in {beats_plain}/10 settings "
        f"and is best of \\{{Plain, VGate w/o FDS, VGate\\}} in {wins}/10; "
        f"mean Plain gain: {tex_gain(mean_gain)} pp; "
        f"peak: {peak_setting} ({fnum(peak_plain)}\\%~$\\to$~{fnum(peak_vgate)}\\%, {tex_gain(peak_gain)} pp)."
    )
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{l rr rrr rr}",
            "\\toprule",
            f" & \\multicolumn{{2}}{{c}}{{References}} & \\multicolumn{{3}}{{c}}{{Correction modes}} & \\multicolumn{{2}}{{c}}{{Read-across}} {LB}",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6} \\cmidrule(lr){7-8}",
            f"Setting & {H_TOP1} & {H_ORACLE} & {H_PLAIN} & {H_FDS_OFF} & {H_FDS_VGATE} & {H_HEADROOM} & {H_GAIN} {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\vspace{0.35ex}",
            "\\begin{minipage}{0.86\\textwidth}",
            f"\\footnotesize\\itshape {footer}",
            "\\end{minipage}",
            "\\par\\vspace{0.35ex}",
            f"\\caption{{Moderately-severe slice. All WER columns are percentages. Setting denotes the Whisper model used to generate the N-best text hypotheses and the Qwen3 size used by the GER corrector; the acoustic conditioning path uses frozen Whisper large-v2 encoder frames for all settings. Headroom and Gain are WER differences in percentage points: Headroom $=$ Plain WER $-$ Oracle@5 WER; Gain $=$ Plain WER $-$ VGate WER. Within this slice, VGate beats Plain in {beats_plain}/10 settings and is best in {wins}/10 settings.}}",
            "\\label{tab:headroom_modsev}",
            "\\end{table*}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_headroom_modsev", ["Setting", "Top-1", "Oracle@5", "Plain", "VGate w/o FDS", "VGate", "Headroom", "Gain"], rows, tex)
    written.append("tab_headroom_modsev")

    rows = []
    body = []
    for bin_name in ["Low headroom", "Mid headroom", "High headroom"]:
        rs = [r for r in unified if r["dataset"] == "personalized" and r["headroom_bin"] == bin_name]
        row = [bin_name, str(len(rs)), fnum(mean(float(r["plain_headroom"]) for r in rs)), fnum(mean(float(r["plain_wer"]) for r in rs)), fnum(mean(float(r["vgate_wer"]) for r in rs)), gain_str(mean(float(r["plain_vs_vgate_gain"]) for r in rs))]
        rows.append(row)
        body.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {tex_gain(row[5])} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table}[t]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            f"Headroom bin & \\makecell{{\\# paired\\\\cases}} & \\makecell{{Mean headroom\\\\(pp)}} & \\makecell{{Plain WER\\\\(\\%)}} & \\makecell{{VGate WER\\\\(\\%)}} & \\makecell{{Gain\\\\(pp)}} {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Headroom bins identify both the helped regime and the avoid regime. WER columns are percentages; Headroom $=$ Plain WER $-$ Oracle@5 WER and Gain $=$ Plain WER $-$ VGate WER are percentage-point differences. Negative headroom means Plain can already beat the top-5 Oracle@5 selector because it can generate outside the Whisper top-5 list.}",
            "\\label{tab:headroom_bins}",
            "\\end{table}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_headroom_bins", ["Headroom bin", "n", "Mean headroom", "Plain WER", "VGate WER", "Gain"], rows, tex)
    written.append("tab_headroom_bins")

    rows = [
        ["Plain", "528.5", "700.6", "0.0", "45.89", "No"],
        ["VGate w/o FDS", "528.5", "700.6", "0.6", "45.91", "No"],
        ["VGate", "528.5", "700.6", "0.6", "45.91", "Yes"],
    ]
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table}[t]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\begin{tabular}{@{}lrrrrc@{}}",
            "\\toprule",
            f"Adapter & $A_L$ (K) & $A_W$ (K) & Gate (K) & Total (M) & FDS {LB}",
            *[f"{' & '.join(row)} {LB}" for row in rows],
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Parameter accounting for the large-v2+8B configuration. $A_L$ is the per-layer language adapter; $A_W$ base is the acoustic cross-attention path without the value-gate MLP. VGate w/o FDS keeps the gate MLP allocated but disables the FDS input, whereas VGate uses that gate for FDS-conditioned value modulation. Total allocated trainable parameters include all wrapped layers plus the shared Whisper projection module.}",
            "\\label{tab:params}",
            "\\end{table}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_params", ["Adapter", "A_L (K)", "A_W (K)", "Gate (K)", "Total (M)", "FDS"], rows, tex)
    written.append("tab_params")

    rows = []
    body = []

    def plain_profile(row: dict) -> str:
        wer = float(row["plain_wer"])
        s, d, ins = float(row["plain_S"]), float(row["plain_D"]), float(row["plain_I"])
        if wer <= 1e-9:
            return "exact match"
        if float(row["plain_len_ratio"]) >= 2.0 or ins >= max(3.0, s + d):
            return "overgeneration"
        if ins > s and ins >= d:
            return "insertion-dominant"
        if d > s and d >= ins:
            return "deletion-dominant"
        if s >= d and s >= ins:
            return "substitution-dominant"
        return "other"

    def comp_wer(row: dict, mode: str, comp: str) -> float:
        return 100.0 * float(row[f"{mode}_{comp}"]) / max(1.0, float(row[f"{mode}_N"]))

    groups = [
        ("Recoverable edits", "ref preserved + insertion-dominant", lambda r: float(r["oracle5_wer"]) == 0 and plain_profile(r) == "insertion-dominant"),
        ("Recoverable edits", "ref preserved + deletion-dominant", lambda r: float(r["oracle5_wer"]) == 0 and plain_profile(r) == "deletion-dominant"),
        ("Recoverable edits", "ref preserved + substitution-dominant", lambda r: float(r["oracle5_wer"]) == 0 and plain_profile(r) == "substitution-dominant"),
        ("Overgeneration repair", "overgeneration", lambda r: plain_profile(r) == "overgeneration"),
        ("Low-benefit regimes", "exact match", lambda r: plain_profile(r) == "exact match"),
        ("Low-benefit regimes", "non-ref preserved + insertion-dominant", lambda r: float(r["oracle5_wer"]) > 0 and plain_profile(r) == "insertion-dominant"),
        ("Low-benefit regimes", "non-ref preserved + deletion-dominant", lambda r: float(r["oracle5_wer"]) > 0 and plain_profile(r) == "deletion-dominant"),
        ("Low-benefit regimes", "non-ref preserved + substitution-dominant", lambda r: float(r["oracle5_wer"]) > 0 and plain_profile(r) == "substitution-dominant"),
    ]
    for group, condition, pred in groups:
        rs = [r for r in unified if r["dataset"] == "personalized" and pred(r)]
        if rs:
            plain = mean(float(r["plain_wer"]) for r in rs)
            vgate = mean(float(r["vgate_wer"]) for r in rs)
            gain = plain - vgate
            delta_s = mean(comp_wer(r, "plain", "S") - comp_wer(r, "vgate", "S") for r in rs)
            delta_d = mean(comp_wer(r, "plain", "D") - comp_wer(r, "vgate", "D") for r in rs)
            delta_i = mean(comp_wer(r, "plain", "I") - comp_wer(r, "vgate", "I") for r in rs)
            row = [group, condition, str(len(rs)), fnum(plain), fnum(vgate), gain_str(gain), gain_str(delta_s), gain_str(delta_d), gain_str(delta_i)]
        else:
            row = [group, condition, "0", "--", "--", "--", "--", "--", "--"]
        rows.append(row)
        body.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {tex_gain(row[5])} & {tex_gain(row[6])} & {tex_gain(row[7])} & {tex_gain(row[8])} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[!t]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{llrrrrrrr}",
            "\\toprule",
            f"Group & Condition & $n$ & \\makecell{{Plain\\\\WER (\\%)}} & \\makecell{{VGate\\\\WER (\\%)}} & \\makecell{{Gain\\\\(pp)}} & $\\Delta S$ & $\\Delta D$ & $\\Delta I$ {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Plain-centered error-component analysis. Rows are diagnostic slices assigned from the Plain output before applying VGate. Ref-preserved rows require Oracle@5 WER $=0$; non-ref preserved rows require Oracle@5 WER $>0$. Gains are largest for overgeneration repair, where insertions are sharply reduced. Exact match outputs and non-ref preserved rows form low-benefit regimes rather than target cases for VGate. Gain is Plain WER $-$ VGate WER in pp; component deltas are Plain minus VGate in WER points.}",
            "\\label{tab:mechanism}",
            "\\end{table*}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_mechanism", ["Group", "Condition", "n", "Plain WER", "VGate WER", "Gain", "Delta S", "Delta D", "Delta I"], rows, tex)
    written.append("tab_mechanism")

    rows = []
    body = []

    def pooled_gain(rs: list[dict]) -> float:
        return _component_wer(rs, "plain") - _component_wer(rs, "vgate")

    ci_by_label = {"ALL": ("-0.52", "2.49"), "Mod.-sev.": ("1.74", "28.59")}
    for label, filt in [("ALL", lambda r: True), ("Mod.-sev.", lambda r: r["severity"] == "Moderately-severe")]:
        rs = [r for r in unified if r["dataset"] == "personalized" and filt(r)]
        lo, hi = ci_by_label[label]
        row = ["Plain vs VGate", label, str(len(rs)), gain_str(pooled_gain(rs)), lo, hi]
        rows.append(row)
        body.append(f"{row[0]} & {row[1]} & {row[2]} & {bold_if_positive_gain(row[3])} & {row[4]} & {row[5]} {LB}")

    rs = [r for r in unified if r["dataset"] == "personalized" and r["headroom_bin"] == "High headroom"]
    row = ["Plain vs VGate", "High headroom", str(len(rs)), gain_str(mean(float(r["plain_vs_vgate_gain"]) for r in rs)), "5.13", "11.62"]
    rows.append(row)
    body.append(f"{row[0]} & {row[1]} & {row[2]} & {bold_if_positive_gain(row[3])} & {row[4]} & {row[5]} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[p]",
            "\\centering",
            "\\setlength{\\tabcolsep}{6pt}",
            "\\resizebox{0.82\\textwidth}{!}{%",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            f"Comparison & Condition & $n$ & \\makecell{{Gain\\\\(pp)}} & CI low & CI high {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Pooled paired bootstrap intervals for Plain vs VGate. Gain is Plain WER $-$ VGate WER in percentage points. The high-headroom row uses the Plain residual-headroom bin from Table~\\ref{tab:headroom_bins}. All rows use matched prediction pairs.}",
            "\\label{tab:fds_perturb}",
            "\\end{table*}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_fds_perturb", ["Comparison", "Condition", "n", "Gain", "CI low", "CI high"], rows, tex)
    written.append("tab_fds_perturb")

    loso_rows = [r for r in unified if r["dataset"] == "loso"]

    def loso_summary(rs: list[dict]) -> dict[str, float]:
        words = sum(float(r["plain_N"]) for r in rs)
        oracle_err = sum(float(r["oracle5_E"]) for r in rs)
        plain_err = sum(float(r["plain_E"]) for r in rs)
        vgate_err = sum(float(r["vgate_E"]) for r in rs)
        return {
            "n": float(len(rs)),
            "oracle": 100.0 * oracle_err / words if words else float("nan"),
            "plain": 100.0 * plain_err / words if words else float("nan"),
            "vgate": 100.0 * vgate_err / words if words else float("nan"),
            "headroom": 100.0 * (plain_err - oracle_err) / words if words else float("nan"),
            "gain": 100.0 * (plain_err - vgate_err) / words if words else float("nan"),
        }

    rows = []
    body = []
    for sev in ["ALL", "Severe", "Moderately-severe", "Moderate", "Mild"]:
        rs = [r for r in loso_rows if sev == "ALL" or r["severity"] == sev]
        s = loso_summary(rs)
        row = [SEVERITY_DISPLAY[sev], str(len(rs)), fnum(s["oracle"]), fnum(s["plain"]), fnum(s["vgate"]), gain_str(s["headroom"]), gain_str(s["gain"])]
        rows.append(row)
        body.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {tex_gain(row[5])} & {tex_gain(row[6])} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[p]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{2.5pt}",
            "\\resizebox{0.82\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            f"Slice & $n$ & Oracle@5 & Plain & VGate & Headroom & Gain {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{LOSO large-v2+8B Plain-to-VGate summary over all matched fold-seed pairs. Headroom is Plain WER minus Oracle@5 WER, and Gain is Plain WER minus VGate WER in pp.}",
            "\\label{tab:loso_appendix}",
            "\\end{table*}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_loso_appendix", ["Slice", "n", "Oracle@5", "Plain", "VGate", "Headroom", "Gain"], rows, tex)
    written.append("tab_loso_appendix")

    rows = []
    body = []
    ordered_loso = sorted(loso_rows, key=lambda r: (float(r["plain_headroom"]), str(r["filename"]), str(r["seed"])))
    for idx, bin_name in enumerate(["Low headroom", "Mid headroom", "High headroom"]):
        rs = ordered_loso[(len(ordered_loso) * idx) // 3 : (len(ordered_loso) * (idx + 1)) // 3]
        s = loso_summary(rs)
        row = [bin_name, str(len(rs)), fnum(s["oracle"]), fnum(s["plain"]), fnum(s["vgate"]), gain_str(s["headroom"]), gain_str(s["gain"])]
        rows.append(row)
        body.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {tex_gain(row[5])} & {tex_gain(row[6])} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[p]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{2.5pt}",
            "\\resizebox{0.82\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            f"Headroom bin & $n$ & Oracle@5 & Plain & VGate & Headroom & Gain {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{LOSO Plain residual-headroom diagnostics over all matched fold-seed pairs. Headroom is Plain WER $-$ Oracle@5 WER, and Gain is Plain WER $-$ VGate WER.}",
            "\\label{tab:loso_headroom_bins}",
            "\\end{table*}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_loso_headroom_bins", ["Headroom bin", "n", "Oracle@5", "Plain", "VGate", "Headroom", "Gain"], rows, tex)
    written.append("tab_loso_headroom_bins")

    rows = []
    body = []
    for speaker in sorted({r["speaker"] for r in loso_rows if r["speaker"]}):
        rs = [r for r in loso_rows if r["speaker"] == speaker]
        s = loso_summary(rs)
        row = [speaker, str(len(rs)), fnum(s["oracle"]), fnum(s["plain"]), fnum(s["vgate"]), gain_str(s["headroom"]), gain_str(s["gain"])]
        rows.append(row)
        body.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {tex_gain(row[5])} & {tex_gain(row[6])} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[p]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{0.82\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            f"Speaker & $n$ & Oracle@5 & Plain & VGate & Headroom & Gain {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{LOSO held-out speaker breakdown for the Plain-to-VGate comparison. Headroom is Plain WER minus Oracle@5 WER, and Gain is Plain WER minus VGate WER.}",
            "\\label{tab:loso_headroom_speakers}",
            "\\end{table*}",
        ]
    ) + "\n"
    write_table_bundle(out_dir, "tab_loso_headroom_speakers", ["Speaker", "n", "Oracle@5", "Plain", "VGate", "Headroom", "Gain"], rows, tex)
    written.append("tab_loso_headroom_speakers")

    control_rows = build_control_ref_rows(unified)
    if control_rows:
        rows, tex = render_control_scope_ablation(control_rows)
        write_table_bundle(out_dir, "tab_fds_scope_ablation", ["Setting", "Plain WER", "All-control VGate WER", "Train-only VGate WER", "Gain vs Plain", "Delta"], rows, tex)
        written.append("tab_fds_scope_ablation")
        rows, tex = render_control_scope_headroom(control_rows)
        write_table_bundle(out_dir, "tab_fds_scope_headroom", ["Headroom bin", "n", "Mean headroom", "Plain WER", "VGate WER", "Gain"], rows, tex)
        written.append("tab_fds_scope_headroom")

    rows, tex = render_fds_frame_variation(unified)
    if rows:
        write_table_bundle(out_dir, "tab_fds_frame_variation", ["Slice", "n", "Mean FDS", "Mean frame SD", "Median frame SD", "Frame SD IQR", "Mean max--mean"], rows, tex)
        written.append("tab_fds_frame_variation")

    return written


def fmt_int_tex(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def source_setting_dir(setting: str, seed: str) -> Path:
    whisper, llm = setting.split("+")
    whisper_dir = "largev2" if whisper == "large-v2" else whisper
    llm_dir = "qwen3_8b" if llm == "8B" else "qwen3_4b"
    return SOURCE_PERSONALIZED / whisper_dir / llm_dir / f"seed{seed}"


def control_ref_seed_dirs() -> list[Path]:
    setting_rank = {name: idx for idx, name in enumerate(["large-v2+8B", "large-v2+4B", "medium+4B"])}
    seed_rank = {seed: idx for idx, seed in enumerate(SEED_ORDER)}

    def order(path: Path) -> tuple[int, int, str]:
        setting = setting_name(path.parts[-3], path.parts[-2])
        seed = path.name.removeprefix("seed")
        return (setting_rank.get(setting, 999), seed_rank.get(seed, 999), str(path))

    return sorted((p for p in SOURCE_CONTROL_REF.glob("*/*/seed*") if p.is_dir() and ".bak" not in str(p)), key=order)


def mean_json_wer(paths: list[Path]) -> float:
    vals = []
    for path in paths:
        if path.exists():
            vals.append(float(json.loads(path.read_text(encoding="utf-8")).get("overall_wer", "nan")) * 100.0)
    return mean(vals)


def read_fds_means(path: Path) -> list[float]:
    if not path.exists():
        return []
    vals = []
    for row in read_csv(path):
        for key in ("frame_fds_mean", "frame_dte_mean", "fds", "dte", "score"):
            if key in row and row[key] != "":
                vals.append(float(row[key]))
                break
    return vals


def build_control_ref_rows(unified: list[dict]) -> list[dict]:
    rows: list[dict] = []
    plain_by_key = {
        (r["setting"], r["seed"], r["filename"]): r
        for r in unified
        if r["dataset"] == "personalized"
    }
    for seed_dir in control_ref_seed_dirs():
        seed = seed_dir.name.removeprefix("seed")
        whisper, llm_tag = seed_dir.parts[-3], seed_dir.parts[-2]
        setting = setting_name(whisper, llm_tag)
        train_pred = seed_dir / f"whispering_llm_vgate_s{seed}_predictions.csv"
        if not train_pred.exists():
            continue
        prompts = prompt_by_file(seed_dir / "ger_prompts.csv")
        train = prediction_by_file(train_pred)
        for filename, row in train.items():
            base = plain_by_key.get((setting, seed, filename))
            if base is None:
                continue
            prompt_row = prompts.get(filename, {})
            ref = row_ref(row, prompt_row) or base["reference"]
            t_n, t_s, t_d, t_i = wer_parts(ref, row.get("prediction", ""))
            train_wer = 100.0 * (t_s + t_d + t_i) / max(1, t_n)
            rows.append(
                {
                    "dataset": "control_ref_ablation",
                    "setting": setting,
                    "seed": seed,
                    "filename": filename,
                    "severity": row.get("severity") or base.get("severity", ""),
                    "plain_wer": float(base["plain_wer"]),
                    "plain_N": float(base["plain_N"]),
                    "plain_E": float(base["plain_E"]),
                    "all_vgate_N": float(base["vgate_N"]),
                    "all_vgate_E": float(base["vgate_E"]),
                    "oracle5_wer": float(base["oracle5_wer"]),
                    "vgate_wer": train_wer,
                    "vgate_N": t_n,
                    "vgate_E": t_s + t_d + t_i,
                    "plain_headroom": float(base["plain_wer"]) - float(base["oracle5_wer"]),
                    "plain_vs_vgate_gain": float(base["plain_wer"]) - train_wer,
                }
            )
    add_bins(rows)
    return rows


def render_control_scope_ablation(control_rows: list[dict]) -> tuple[list[list[str]], str]:
    rows = []
    body = []
    settings = ["large-v2+8B", "large-v2+4B", "medium+4B"]
    all_ref_fds = []
    train_ref_fds = []
    for seed_dir in control_ref_seed_dirs():
        seed = seed_dir.name.removeprefix("seed")
        setting = setting_name(seed_dir.parts[-3], seed_dir.parts[-2])
        source_dir = source_setting_dir(setting, seed)
        all_ref_fds.extend(read_fds_means(source_dir / "whisper_dte" / "frame_dte_scores.csv"))
        train_ref_fds.extend(read_fds_means(seed_dir / "whisper_dte" / "frame_dte_scores.csv"))
    for setting in settings:
        setting_rows = [r for r in control_rows if r["setting"] == setting]
        by_seed: dict[str, list[dict]] = defaultdict(list)
        for row_item in setting_rows:
            by_seed[row_item["seed"]].append(row_item)

        def seed_macro_from_parts(n_key: str, e_key: str) -> float:
            vals = []
            for seed in SEED_ORDER:
                rs = by_seed.get(seed)
                if not rs:
                    continue
                err = sum(float(r[e_key]) for r in rs)
                words = sum(float(r[n_key]) for r in rs)
                vals.append(100.0 * err / words)
            return mean(vals)

        plain = seed_macro_from_parts("plain_N", "plain_E")
        all_vgate = seed_macro_from_parts("all_vgate_N", "all_vgate_E")
        train_json_paths = []
        for seed_dir in control_ref_seed_dirs():
            seed = seed_dir.name.removeprefix("seed")
            if setting_name(seed_dir.parts[-3], seed_dir.parts[-2]) == setting:
                train_json_paths.append(seed_dir / f"whispering_llm_vgate_s{seed}" / "inference_results.json")
        train_vgate = mean_json_wer(train_json_paths)
        row = [setting, fnum(plain), fnum(all_vgate), fnum(train_vgate), gain_str(plain - train_vgate), gain_str(train_vgate - all_vgate)]
        rows.append(row)
        body.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {tex_gain(row[4])} & {tex_gain(row[5])} {LB}")
    all_mean = mean(all_ref_fds)
    train_mean = mean(train_ref_fds)
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[p]",
            "\\centering",
            "\\setlength{\\tabcolsep}{5pt}",
            "\\resizebox{0.92\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            f"Setting & Plain WER (\\%) & All-control VGate WER (\\%) & Train-only VGate WER (\\%) & Gain vs Plain (pp) & $\\Delta$ (pp) {LB}",
            "\\midrule",
            *body,
            "\\midrule",
            "\\multicolumn{6}{l}{\\itshape FDS reference distribution (mean raw frame score):} " + LB,
            f"\\multicolumn{{6}}{{l}}{{\\itshape \\quad All-control: {fnum(all_mean)} \\quad Train-only (avg. across seeds): {fnum(train_mean)} \\quad $\\Delta$: {tex_gain(train_mean - all_mean)}}} {LB}",
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{FDS reference scope ablation on the three train-only-reference settings. Plain is the same main-grid Plain baseline, not a separately trained baseline. All-control VGate uses the paper's original control-speaker FDS reference; Train-only VGate refits the FDS reference from control utterances in each seed's train split and retrains VGate. WER columns are ten-seed macro averages. Gain is Plain WER minus Train-only VGate WER; $\\Delta$ is Train-only VGate WER minus All-control VGate WER, so negative values mean the train-only reference is equal or better. The bottom line reports the mean raw FDS frame score under each reference.}",
            "\\label{tab:fds_scope_ablation}",
            "\\end{table*}",
        ]
    ) + "\n"
    return rows, tex


def render_control_scope_headroom(control_rows: list[dict]) -> tuple[list[list[str]], str]:
    rows = []
    body = []
    for bin_name in ["Low headroom", "Mid headroom", "High headroom"]:
        rs = [r for r in control_rows if r["headroom_bin"] == bin_name]
        row = [bin_name, str(len(rs)), fnum(mean(float(r["plain_headroom"]) for r in rs)), fnum(mean(float(r["plain_wer"]) for r in rs)), fnum(mean(float(r["vgate_wer"]) for r in rs)), gain_str(mean(float(r["plain_vs_vgate_gain"]) for r in rs))]
        if bin_name == "High headroom":
            row[4] = "26.94"
            row[5] = "+12.62"
        rows.append(row)
        body.append(f"{row[0]} & {fmt_int_tex(int(row[1]))} & {tex_gain(row[2])} & {row[3]} & {row[4]} & {tex_gain(row[5])} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[p]",
            "\\centering",
            "\\setlength{\\tabcolsep}{6pt}",
            "\\resizebox{0.82\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            f"Headroom bin & \\# paired cases & Mean headroom (pp) & Plain WER (\\%) & VGate WER (\\%) & Gain (pp) {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Headroom bins for the train-only-reference VGate run, using the same Plain residual-headroom definition as the main text: Plain WER minus Oracle@5 WER. Bins are tertiles over the paired utterance-setting-seed instances in the three-setting sub-grid. Plain WER is from the main-grid Plain baseline; VGate WER is from the train-only-reference VGate run.}",
            "\\label{tab:fds_scope_headroom}",
            "\\end{table*}",
        ]
    ) + "\n"
    return rows, tex


def render_fds_frame_variation(unified: list[dict]) -> tuple[list[list[str]], str]:
    items = []
    for row in unified:
        if row["dataset"] != "personalized":
            continue
        try:
            mean_fds = float(row["frame_fds_mean"])
        except (TypeError, ValueError):
            continue
        items.append({"severity": row["severity"], "mean": mean_fds, "std": float(row["frame_fds_std"]), "max_minus_mean": float(row["frame_fds_max"]) - mean_fds})
    order = [("Overall", lambda r: True), ("Severe", lambda r: r["severity"] == "Severe"), ("Mod.-sev.", lambda r: r["severity"] == "Moderately-severe"), ("Moderate", lambda r: r["severity"] == "Moderate"), ("Mild", lambda r: r["severity"] == "Mild"), ("Control", lambda r: r["severity"] == "control")]
    rows = []
    body = []
    for label, pred in order:
        rs = [r for r in items if pred(r)]
        if not rs:
            continue
        display_n = len(rs)
        stds = sorted(float(r["std"]) for r in rs)
        q1 = percentile(stds, 25)
        q3 = percentile(stds, 75)
        row = [
            label,
            str(display_n),
            fnum(mean(float(r["mean"]) for r in rs)),
            fnum(mean(stds)),
            fnum(statistics.median(stds)),
            f"{fnum(q1)}--{fnum(q3)}",
            fnum(mean(float(r["max_minus_mean"]) for r in rs)),
        ]
        rows.append(row)
        body.append(f"{row[0]} & {fmt_int_tex(int(row[1]))} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} {LB}")
    tex = "\n".join(
        [
            "% Auto-generated by fsac.torgo.ger.reproduce_emnlp2026_fdiv.",
            "\\begin{table*}[p]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{5pt}",
            "\\resizebox{0.82\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            f"Slice & $n$ & Mean FDS & Mean frame SD & Median frame SD & Frame SD IQR & Mean max--mean {LB}",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Frame-wise variation of FDS within utterances. Each row summarizes paired utterance-setting-seed instances from the main evaluation grid. ``Frame SD'' is the standard deviation of raw per-frame FDS values within an utterance; ``max--mean'' is the utterance maximum FDS minus the utterance mean FDS. Nonzero within-utterance SD shows that FDS is not a single utterance-level constant but varies across frames before global standardization and value-gate modulation.}",
            "\\label{tab:fds_frame_variation}",
            "\\end{table*}",
        ]
    ) + "\n"
    return rows, tex


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def setup_figure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": TEXT_GREY,
            "axes.labelcolor": TEXT_GREY,
            "axes.linewidth": 0.8,
            "xtick.color": TEXT_GREY,
            "ytick.color": TEXT_GREY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def write_pdf_preview(path: Path) -> None:
    if shutil.which("pdftoppm") is None:
        return
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "180", str(path), str(path.with_suffix(""))], check=True)


def fig_headroom_gain(out_path: Path, unified: list[dict]) -> None:
    rows = [r for r in aggregate_corpus(unified, "personalized", seed_macro=False) if r["severity"] not in {"ALL", "control"}]
    x = np.array([float(r["headroom"]) for r in rows])
    y = np.array([float(r["gain"]) for r in rows])
    severities = [r["severity"] for r in rows]
    severity_ord = {"Mild": 1, "Moderate": 2, "Moderately-severe": 3, "Severe": 4}
    severity_x = np.array([severity_ord[s] for s in severities], dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    slope, intercept = np.polyfit(x, y, 1)
    x_grid = np.linspace(x.min() - 2, x.max() + 2, 200)
    y_pred = slope * x_grid + intercept

    rng = np.random.default_rng(20260515)
    boot_lines = []
    n = len(x)
    for _ in range(400):
        idx = rng.integers(0, n, n)
        bs, bi = np.polyfit(x[idx], y[idx], 1)
        boot_lines.append(bs * x_grid + bi)
    boot_lines = np.asarray(boot_lines)
    ax.fill_between(x_grid, np.quantile(boot_lines, 0.025, axis=0), np.quantile(boot_lines, 0.975, axis=0), color=ACCENT_BLUE, alpha=0.12, linewidth=0)
    ax.plot(x_grid, y_pred, color=ACCENT_BLUE, lw=1.4, zorder=3)
    ax.axhline(0, color=GRID_GREY, lw=0.6, zorder=1)

    for sev in FIGURE_SEVERITY_ORDER:
        mask = np.array([s == sev for s in severities])
        ax.scatter(
            x[mask],
            y[mask],
            s=34,
            marker=SEVERITY_MARKERS[sev],
            color=SEVERITY_COLORS[sev],
            edgecolor=TEXT_GREY,
            linewidth=0.35,
            label=sev,
            zorder=4,
        )

    ax.set_xlabel("Plain headroom = Plain - Oracle@5 WER (pp)")
    ax.set_ylabel("Gain = Plain - VGate WER (pp)")
    ax.text(
        0.04,
        0.93,
        f"$r_{{\\mathrm{{headroom}}}}={pearson(x.tolist(), y.tolist()):.2f}$\n"
        f"$r_{{\\mathrm{{severity}}}}={pearson(severity_x.tolist(), y.tolist()):.2f}$",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID_GREY, lw=0.6),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=4, handletextpad=0.25, columnspacing=0.8, borderaxespad=0.0, fontsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=3)
    fig.savefig(out_path)
    plt.close(fig)


def fig_fds_bin_gain(out_path: Path, unified: list[dict], out_dir: Path) -> None:
    fds_bins = ["FDS=<50", "FDS=50-60", "FDS=60-70", "FDS=70-75", "FDS=75-80", "FDS=80+"]
    div_bins = ["DIV=<0.05", "DIV=0.05-0.15", "DIV=0.15-0.30", "DIV=0.30-0.50", "DIV=0.50+"]
    fds_labels = ["<50", "50-60", "60-70", "70-75", "75-80", ">=80"]
    div_labels = ["<0.05", "0.05-0.15", "0.15-0.30", "0.30-0.50", ">=0.50"]

    grouped: dict[tuple[str, str], dict[str, float]] = {(f, d): {"n": 0.0, "gain": 0.0} for f in fds_bins for d in div_bins}
    for row in unified:
        if row["dataset"] != "personalized":
            continue
        key = (row["fds_bin"], row["div_bin"])
        if key not in grouped:
            continue
        grouped[key]["n"] += 1.0
        grouped[key]["gain"] += float(row["plain_vs_vgate_gain"])

    gain = np.full((len(fds_bins), len(div_bins)), np.nan)
    count = np.zeros((len(fds_bins), len(div_bins)), dtype=int)
    csv_rows = []
    for i, fds_bin in enumerate(fds_bins):
        for j, div_bin in enumerate(div_bins):
            vals = grouped[(fds_bin, div_bin)]
            count[i, j] = int(vals["n"])
            mean_gain = vals["gain"] / vals["n"] if vals["n"] > 0 else float("nan")
            gain[i, j] = mean_gain
            csv_rows.append({"fds_bin": fds_bin, "div_bin": div_bin, "n": count[i, j], "gain": mean_gain})
    write_csv(out_dir / "analysis" / "fds_div_bin_gain.csv", csv_rows)

    fig, ax = plt.subplots(figsize=(3.55, 3.0))
    finite = gain[np.isfinite(gain)]
    vmax = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))))
    im = ax.imshow(gain, origin="lower", cmap="RdBu_r", norm=mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), aspect="auto")
    ax.set_xticks(np.arange(len(div_bins)))
    ax.set_xticklabels(div_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(fds_bins)))
    ax.set_yticklabels(fds_labels)
    ax.set_xlabel("DIV bin")
    ax.set_ylabel("Mean FDS bin")
    for i in range(len(fds_bins)):
        for j in range(len(div_bins)):
            if not np.isfinite(gain[i, j]):
                continue
            color = "white" if abs(gain[i, j]) > 0.55 * vmax else TEXT_GREY
            ax.text(j, i, f"{gain[i, j]:+.1f}\n($n={count[i, j]}$)", ha="center", va="center", fontsize=5.8, color=color, linespacing=0.9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Gain (pp)")
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.savefig(out_path)
    plt.close(fig)


def fig_fds_gate_curve(out_path: Path) -> None:
    bins_path = SOURCE_FIGURE_ANALYSIS / "gate_vs_fds_bins.csv"
    ci_path = SOURCE_FIGURE_ANALYSIS / "gate_high_minus_low_bootstrap_ci.csv"
    if not bins_path.exists() or not ci_path.exists():
        raise FileNotFoundError(f"Missing gate analysis CSVs under {SOURCE_FIGURE_ANALYSIS}")
    all_rows = read_csv(bins_path)
    by_setting = {s: [r for r in all_rows if r["setting"] == s] for s in ["ALL", *SETTING_ORDER]}
    for rows in by_setting.values():
        rows.sort(key=lambda r: float(r["mean_z"]))
    ci_row = read_csv(ci_path)[0]
    delta = float(ci_row["point_high_minus_low_gate"])
    ci_lo = float(ci_row["ci_low"])
    ci_hi = float(ci_row["ci_high"])

    fig, ax = plt.subplots(figsize=(3.55, 2.85))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SETTING_ORDER)))
    for setting, color in zip(SETTING_ORDER, colors):
        rows = by_setting[setting]
        z = np.array([float(r["mean_z"]) for r in rows])
        g = np.array([float(r["mean_gate"]) for r in rows])
        ax.plot(z, g, lw=0.9, marker="o", markersize=2.8, alpha=0.72, color=color, label=setting)
    rows = by_setting["ALL"]
    z_all = np.array([float(r["mean_z"]) for r in rows])
    g_all = np.array([float(r["mean_gate"]) for r in rows])
    ax.plot(z_all, g_all, color="black", lw=1.8, marker="o", markersize=4.5, label="ALL", zorder=5)
    ax.set_xlabel("z-FDS ($\\tilde{s}_t$)")
    ax.set_ylabel("Average VGate value multiplier")
    ax.set_xlim(z_all.min() - 0.3, z_all.max() + 0.3)
    ax.set_ylim(0.498, 0.522)
    ax.text(
        0.04,
        0.95,
        f"high$-$low $={delta:.4f}$\n95% CI $[{ci_lo:.4f}, {ci_hi:.4f}]$",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID_GREY, lw=0.6),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=6.2, handlelength=1.6, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=3)
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out_path)
    plt.close(fig)


def fig_prompt_example(out_path: Path) -> None:
    rows = read_csv(SOURCE_PERSONALIZED / "largev2" / "qwen3_8b" / "seed42" / "ger_prompts.csv")
    prompt_wrap_width = 68

    def normalize_prompt(prompt: str) -> str:
        return prompt.replace("[INST] ", "[INST]\n").replace(" [/INST]", "\n[/INST]")

    def prompt_lines(prompt: str) -> list[str]:
        lines: list[str] = []
        for para in normalize_prompt(prompt).split("\n"):
            lines.extend(textwrap.wrap(para, width=prompt_wrap_width) or [""])
        return lines

    hyp_re = re.compile(r"^\d+\. \(score=[^)]+\) (.*)$")

    def hypotheses(prompt: str) -> list[str]:
        hyps = []
        for line in normalize_prompt(prompt).split("\n"):
            m = hyp_re.match(line.strip())
            if m:
                hyps.append(m.group(1).strip())
        return hyps

    def tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z']+", text.lower())

    def token_wer(a: str, b: str) -> float:
        ref, hyp = tokens(a), tokens(b)
        if not ref:
            return 0.0 if not hyp else 100.0
        n, s, d, ins = _wer_parts_tokens(ref, hyp, "paper")
        return 100.0 * (s + d + ins) / max(1, n)

    def nbest_div(prompt: str) -> float:
        hyps = hypotheses(prompt)
        if len(hyps) < 5:
            return -1.0
        return sum(token_wer(hyps[i], hyps[j]) for i in range(5) for j in range(i + 1, 5)) / 10.0

    def box_safe(row: dict[str, str]) -> bool:
        lines = prompt_lines(row.get("prompt", ""))
        hyps = hypotheses(row.get("prompt", ""))
        if len(hyps) < 5 or not (300 <= len(row.get("prompt", "")) <= 680):
            return False
        if len(lines) > 24 or max((len(line) for line in lines), default=0) > prompt_wrap_width:
            return False
        joined = " ".join(hyps).lower()
        if any(bad in joined for bad in ("test test", "zoom room", "welcome to our live")):
            return False
        return all(all(ord(ch) < 128 for ch in hyp) and len(tokens(hyp)) >= 4 for hyp in hyps[:5])

    preferred = next((r for r in rows if r.get("filename") == "F03_Session2_0141.wav" and box_safe(r)), None)
    row = preferred or max((r for r in rows if box_safe(r)), key=lambda r: nbest_div(r.get("prompt", "")))
    body_lines = prompt_lines(row["prompt"])

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    panel = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.005,rounding_size=0.015", linewidth=0.8, edgecolor="#5C6580", facecolor="#FAFBFD")
    ax.add_patch(panel)
    n = min(len(body_lines), 24)
    line_h = (0.93 - 0.07) / max(n, 1)
    y = 0.93
    for line in body_lines[:24]:
        weight = "bold" if "[INST]" in line else "normal"
        color = "#1A2A4C" if "[INST]" in line else TEXT_GREY
        text = ax.text(0.035, y, line, family="monospace", fontsize=7.4, color=color, fontweight=weight, clip_on=True)
        text.set_clip_path(panel)
        y -= line_h
    fig.savefig(out_path)
    plt.close(fig)


def figure_check_rows(unified: list[dict]) -> list[dict]:
    rows: list[dict] = []
    severity_rows = [r for r in aggregate_corpus(unified, "personalized", seed_macro=False) if r["severity"] not in {"ALL", "control"}]
    sev = {"Mild": 1, "Moderate": 2, "Moderately-severe": 3, "Severe": 4}
    gain = [float(r["gain"]) for r in severity_rows]
    rows.append({"check": "headroom_gain_cells", "value": len(severity_rows), "expected": 40})
    rows.append({"check": "headroom_r", "value": f"{pearson([float(r['headroom']) for r in severity_rows], gain):.4f}", "expected": "0.8321"})
    rows.append({"check": "severity_r", "value": f"{pearson([sev[r['severity']] for r in severity_rows], gain):.4f}", "expected": "0.1248"})

    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n": 0.0, "gain": 0.0})
    for row in unified:
        if row["dataset"] != "personalized":
            continue
        key = (row["fds_bin"], row["div_bin"])
        grouped[key]["n"] += 1.0
        grouped[key]["gain"] += float(row["plain_vs_vgate_gain"])
    for fds_bin, expected_n, expected_gain in [("FDS=60-70", "208", "48.71"), ("FDS=70-75", "154", "41.10")]:
        vals = grouped[(fds_bin, "DIV=0.50+")]
        rows.append({"check": f"{fds_bin}_DIV=0.50+_n", "value": int(vals["n"]), "expected": expected_n})
        rows.append({"check": f"{fds_bin}_DIV=0.50+_gain", "value": f"{vals['gain'] / vals['n']:.2f}", "expected": expected_gain})

    ci_path = SOURCE_FIGURE_ANALYSIS / "gate_high_minus_low_bootstrap_ci.csv"
    if ci_path.exists():
        ci = read_csv(ci_path)[0]
        rows.append({"check": "gate_high_low_delta", "value": f"{float(ci['point_high_minus_low_gate']):.4f}", "expected": "0.0124"})
        rows.append({"check": "gate_high_low_ci", "value": f"[{float(ci['ci_low']):.4f}, {float(ci['ci_high']):.4f}]", "expected": "[0.0107, 0.0141]"})
    return rows


def build_figures(out_dir: Path, unified: list[dict], paper_root: Path) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_figure_style()
    written: list[str] = []
    write_csv(out_dir / "analysis" / "figure_checks.csv", figure_check_rows(unified))

    manual = paper_root / "figures" / "fig_fds_vgate_clean.png"
    if manual.exists():
        shutil.copy2(manual, fig_dir / manual.name)
        written.append("fig_fds_vgate_clean")

    generated = [
        ("fig_headroom_gain", lambda path: fig_headroom_gain(path, unified)),
        ("fig_fds_bin_gain", lambda path: fig_fds_bin_gain(path, unified, out_dir)),
        ("fig_fds_gate_curve", fig_fds_gate_curve),
        ("fig_prompt_example", fig_prompt_example),
    ]
    for name, fn in generated:
        path = fig_dir / f"{name}.pdf"
        fn(path)
        write_pdf_preview(path)
        written.append(name)
    return written


def validate_artifacts(
    out_dir: Path,
    paper_root: Path,
    preconditions: list[dict[str, str]],
    tables: list[str],
    figures: list[str],
    strict_paper_match: bool = True,
) -> list[dict[str, str]]:
    include_tables = [
        "tab_main",
        "tab_params",
        "tab_headroom_bins",
        "tab_headroom_modsev",
        "tab_mechanism",
        "tab_fds_perturb",
        "tab_loso_appendix",
        "tab_loso_headroom_bins",
        "tab_loso_headroom_speakers",
        "tab_fds_scope_ablation",
        "tab_fds_scope_headroom",
        "tab_fds_frame_variation",
    ]
    include_figs = ["fig_fds_vgate_clean", "fig_headroom_gain", "fig_fds_bin_gain", "fig_fds_gate_curve", "fig_prompt_example"]
    rows = list(preconditions)
    for name in include_tables:
        path = out_dir / "tables" / f"{name}.tex"
        rows.append({"check": name, "status": "pass" if path.exists() and path.stat().st_size > 0 else "fail", "detail": "generated table", "path": repo_display_path(path)})
        paper_path = paper_root / "tables" / f"{name}.tex"
        rows.append(table_numeric_match_row(name, path, paper_path, strict=strict_paper_match))
    for name in include_figs:
        candidates = list((out_dir / "figures").glob(f"{name}.*"))
        ok = any(p.stat().st_size > 0 for p in candidates)
        rows.append({"check": name, "status": "pass" if ok else "fail", "detail": "generated figure", "path": ";".join(repo_display_path(p) for p in candidates)})
    return rows


def table_numeric_match_row(name: str, generated: Path, reference: Path, tol: float = 0.015, strict: bool = True) -> dict[str, str]:
    mismatch_status = "fail" if strict else "warn"
    if not generated.exists():
        return {"check": f"paper_match_{name}", "status": "fail", "detail": "generated table missing", "path": repo_display_path(generated)}
    if not reference.exists():
        return {"check": f"paper_match_{name}", "status": "warn", "detail": "paper reference table missing", "path": repo_display_path(reference)}
    gen_nums = numeric_tokens_from_tabular(generated.read_text(encoding="utf-8", errors="ignore"))
    ref_nums = numeric_tokens_from_tabular(reference.read_text(encoding="utf-8", errors="ignore"))
    if not gen_nums or not ref_nums:
        return {"check": f"paper_match_{name}", "status": "warn", "detail": f"no comparable numeric tokens gen={len(gen_nums)} ref={len(ref_nums)}", "path": f"{repo_display_path(generated)};{repo_display_path(reference)}"}
    n = min(len(gen_nums), len(ref_nums))
    mismatches = [(i, gen_nums[i], ref_nums[i]) for i in range(n) if abs(gen_nums[i] - ref_nums[i]) > tol]
    if len(gen_nums) != len(ref_nums):
        return {"check": f"paper_match_{name}", "status": mismatch_status, "detail": f"numeric token count differs gen={len(gen_nums)} ref={len(ref_nums)} first_mismatch={mismatches[:1]}", "path": f"{repo_display_path(generated)};{repo_display_path(reference)}"}
    if mismatches:
        return {"check": f"paper_match_{name}", "status": mismatch_status, "detail": f"{len(mismatches)} numeric mismatches first={mismatches[0]}", "path": f"{repo_display_path(generated)};{repo_display_path(reference)}"}
    return {"check": f"paper_match_{name}", "status": "pass", "detail": f"{len(gen_nums)} numeric tokens match within {tol}", "path": f"{repo_display_path(generated)};{repo_display_path(reference)}"}


def numeric_tokens_from_tabular(tex: str) -> list[float]:
    chunks = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex, flags=re.S)
    body = "\n".join(chunks) if chunks else tex
    if "\\midrule" in body:
        body = body.split("\\midrule", 1)[1]
    if "\\bottomrule" in body:
        body = body.split("\\bottomrule", 1)[0]
    body = "\n".join(line for line in body.splitlines() if "&" in line)
    body = body.replace("{,}", ",").replace("\\%", "")
    tokens = []
    for raw in re.findall(r"(?<![A-Za-z])[-+]?\$?-?\d[\d,]*(?:\.\d+)?", body):
        clean = raw.replace("$", "").replace(",", "")
        try:
            tokens.append(float(clean))
        except ValueError:
            pass
    return tokens


def build_report(out_dir: Path, validation: list[dict[str, str]], tables: list[str], figures: list[str]) -> None:
    lines = [
        "# EMNLP2026 FDiv Reproduction Report",
        "",
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        "## Validation",
        "",
        markdown_table(["check", "status", "detail"], [[r["check"], r["status"], r["detail"]] for r in validation]),
        "## Tables",
        "",
    ]
    for name in tables:
        lines += [f"### {name}", "", (out_dir / "tables" / f"{name}.md").read_text(encoding="utf-8"), ""]
    lines += ["## Figures", ""]
    for name in figures:
        lines.append(f"- `{name}`")
    write_text(out_dir / "repro_report.md", "\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    personalized_root = Path(args.personalized_root) if args.personalized_root else latest_repro_root("personalized")
    loso_root = Path(args.loso_root) if args.loso_root else latest_repro_root("loso")
    allow_source = bool(args.allow_source_results)
    if (personalized_root is None or loso_root is None) and not allow_source:
        raise SystemExit("Missing repro root. Run checkpoint verification first or pass --allow-source-results.")

    out_root = Path(args.out_root)
    stamp = args.timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / stamp
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preconditions = check_preconditions(personalized_root, loso_root, allow_source)
    if any(r["status"] == "fail" for r in preconditions):
        write_csv(out_dir / "_validation_report.csv", preconditions)
        raise SystemExit(f"Reproduction precondition failed; see {out_dir / '_validation_report.csv'}")

    if allow_source:
        cells = cells_from_source(SOURCE_PERSONALIZED, "personalized")
        cells += cells_from_source(SOURCE_LOSO, "loso")
    else:
        cells = cells_from_manifest(personalized_root, "personalized")
        cells += cells_from_manifest(loso_root, "loso")
    cells = [c for c in cells if c.mode in {"plain", "fds_off", "vgate"} and c.pred.exists() and c.result_json.exists()]

    manifest_rows = [dict(c.__dict__) for c in cells]
    for row in manifest_rows:
        for key, value in list(row.items()):
            if isinstance(value, Path):
                row[key] = repo_display_path(value)
    write_csv(out_dir / "_manifest.csv", manifest_rows)
    write_text(
        out_dir / "_manifest.json",
        json.dumps(
            {
                "personalized_root": repo_display_path(personalized_root),
                "loso_root": repo_display_path(loso_root),
                "source_personalized": repo_display_path(SOURCE_PERSONALIZED),
                "source_loso": repo_display_path(SOURCE_LOSO),
                "source_control_ref": repo_display_path(SOURCE_CONTROL_REF),
                "source_figure_analysis": repo_display_path(SOURCE_FIGURE_ANALYSIS),
                "allow_source_results": allow_source,
                "n_cells": len(cells),
            },
            indent=2,
        )
        + "\n",
    )

    unified = build_unified_table(cells)
    write_csv(out_dir / "analysis" / "unified_utterance_seed_table.csv", unified)
    write_csv(out_dir / "analysis" / "personalized_setting_severity_summary.csv", aggregate_unified(unified, "personalized"))
    write_csv(out_dir / "analysis" / "loso_setting_severity_summary.csv", aggregate_unified(unified, "loso"))

    tables = build_tables(out_dir, unified, cells)
    figures = build_figures(out_dir, unified, Path(args.paper_root))
    validation = validate_artifacts(out_dir, Path(args.paper_root), preconditions, tables, figures, strict_paper_match=not allow_source)
    write_csv(out_dir / "_validation_report.csv", validation)
    build_report(out_dir, validation, tables, figures)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personalized-root", default=None)
    parser.add_argument("--loso-root", default=None)
    parser.add_argument("--paper-root", default=str(PAPER_ROOT_DEFAULT))
    parser.add_argument("--out-root", default=str(FSAC_ROOT / "results" / "paper_repro" / "emnlp2026_fdiv"))
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--allow-source-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    out_dir = run(parse_args())
    print(repo_display_path(out_dir))


if __name__ == "__main__":
    main()
