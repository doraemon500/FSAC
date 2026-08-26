from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]



DEFAULT_PAIRS = [
    ("largev2", "qwen3_8b"),
    ("largev2", "qwen3_4b"),
    ("medium", "qwen3_4b"),
]
DEFAULT_SEEDS = [42, 123, 456, 789, 999, 2024, 2025, 3407, 777, 888]


SEVERITY_KEYS = ["Severe", "Moderately-severe", "Moderate", "Mild", "control"]
HIGH_HEADROOM_BIN = "Moderately-severe"



def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _per_seed_metrics(payload: dict) -> dict:

    sev_in = payload.get("severity_breakdown", {}) or {}
    sev = {}
    for k, v in sev_in.items():
        if isinstance(v, dict) and "wer" in v:
            sev[k] = float(v["wer"])
    return {
        "overall_wer": float(payload.get("overall_wer", float("nan"))),
        "overall_cer": float(payload.get("overall_cer", float("nan"))),
        "severity": sev,
        "n_samples": int(payload.get("n_samples", 0) or 0),
    }


def _setting_a_paths(whisper: str, llm_tag: str, seed: int, mode: str, root: Path) -> Path:
    paper_mode = {"vgate": "vgate", "fds_off": "dte_off", "plain": "plain"}.get(mode, mode)
    cell = root / whisper / llm_tag / f"seed{seed}"
    if mode == "plain":
        return cell / f"whispering_llama_{paper_mode}_s{seed}" / "inference_results.json"
    return cell / f"whispering_llama_{paper_mode}_s{seed}" / "inference_results.json"


def _setting_b_path(whisper: str, llm_tag: str, seed: int, root: Path) -> Path:
    return (
        root
        / whisper
        / llm_tag
        / f"seed{seed}"
        / f"whispering_llm_vgate_s{seed}"
        / "inference_results.json"
    )


def _setting_b_corr_path(whisper: str, llm_tag: str, seed: int, root: Path) -> Path:
    return (
        root
        / whisper
        / llm_tag
        / f"seed{seed}"
        / f"whispering_llm_vgate_s{seed}"
        / "gate_fds_corr.json"
    )



def _load_paper_reference_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _plain_aggregate_from_reference(
    df: pd.DataFrame, whisper: str, llm_tag: str
) -> dict | None:

    if df.empty:
        return None
    rows = df[
        (df["whisper"] == whisper)
        & (df["llm_tag"] == llm_tag)
        & (df["mode"] == "plain")
    ]
    if rows.empty:
        return None
    out: dict = {"source": "paper_table_reference.csv", "n_seeds": None, "severity": {}}
    overall = rows[rows["severity"] == "overall"]
    if not overall.empty:
        out["overall_wer_mean_pct"] = float(overall.iloc[0]["wer_mean"])
        out["overall_wer_std_pct"] = float(overall.iloc[0]["wer_std"])
        out["n_seeds"] = int(overall.iloc[0]["n_seeds"])
    for sev in SEVERITY_KEYS:
        srow = rows[rows["severity"] == sev]
        if not srow.empty:
            out["severity"][sev] = {
                "wer_mean_pct": float(srow.iloc[0]["wer_mean"]),
                "wer_std_pct": float(srow.iloc[0]["wer_std"]),
            }
    return out



def _mean_std_pct(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.array([v for v in values if not (v is None or np.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean() * 100.0), float(arr.std() * 100.0), int(arr.size)


def _classify(delta_overall_pp: float, delta_modsev_pp: float, gain_a_pp: float, gain_b_pp: float) -> str:

    if np.isnan(delta_overall_pp) or np.isnan(gain_b_pp):
        return "insufficient_data"
    abs_overall = abs(delta_overall_pp)
    sign_a = np.sign(gain_a_pp) if not np.isnan(gain_a_pp) else 0.0
    sign_b = np.sign(gain_b_pp)
    direction_match = sign_a == sign_b and sign_b != 0
    modsev_positive = (not np.isnan(delta_modsev_pp)) and gain_b_pp >= 0
    if abs_overall <= 1.0:
        return "very_strong (|ΔWER|≤1pp)"
    if direction_match and modsev_positive:
        return "reviewer_defense_success (gain direction + mod.-sev. preserved)"
    if direction_match:
        return "sufficient (core gain direction preserved)"
    if modsev_positive:
        return "acceptable (overall wiggles; mod.-sev./high-headroom preserved)"
    return "trend_collapsed (report sensitivity)"



def _load_raw_score_dist(path: Path) -> tuple[float, float, int]:
    if not path.exists():
        return float("nan"), float("nan"), 0
    data = np.load(str(path), allow_pickle=True)
    fnames = list(data["filenames"])
    chunks = [data[f"scores_{i}"].astype(np.float32) for i in range(len(fnames))]
    if not chunks:
        return float("nan"), float("nan"), 0
    arr = np.concatenate(chunks)
    return float(arr.mean()), float(arr.std()), int(arr.size)



def _per_seed_collect(
    whisper: str,
    llm_tag: str,
    seeds: list[int],
    a_root: Path,
    b_root: Path,
    a_plain_root: Path,
) -> tuple[list[dict], list[dict]]:

    rows: list[dict] = []
    gate_rows: list[dict] = []
    for seed in seeds:
        row = {"whisper": whisper, "llm": llm_tag, "seed": seed}

        a_v = _read_json(_setting_a_paths(whisper, llm_tag, seed, "vgate", a_root))
        if a_v is not None:
            m = _per_seed_metrics(a_v)
            row["A_vgate_overall_wer"] = m["overall_wer"]
            for s in SEVERITY_KEYS:
                row[f"A_vgate_{s}_wer"] = m["severity"].get(s, float("nan"))
        else:
            row["A_vgate_overall_wer"] = float("nan")
            for s in SEVERITY_KEYS:
                row[f"A_vgate_{s}_wer"] = float("nan")

        a_f = _read_json(_setting_a_paths(whisper, llm_tag, seed, "fds_off", a_root))
        if a_f is not None:
            m = _per_seed_metrics(a_f)
            row["A_fds_off_overall_wer"] = m["overall_wer"]
            for s in SEVERITY_KEYS:
                row[f"A_fds_off_{s}_wer"] = m["severity"].get(s, float("nan"))
        else:
            row["A_fds_off_overall_wer"] = float("nan")
            for s in SEVERITY_KEYS:
                row[f"A_fds_off_{s}_wer"] = float("nan")
        plain_payload = _read_json(_setting_a_paths(whisper, llm_tag, seed, "plain", a_plain_root))
        if plain_payload is None:
            plain_p = a_plain_root / whisper / llm_tag / f"seed{seed}" / f"whispering_llm_plain_s{seed}" / "inference_results.json"
            plain_payload = _read_json(plain_p)
        if plain_payload is not None:
            m = _per_seed_metrics(plain_payload)
            row["plain_overall_wer"] = m["overall_wer"]
            for s in SEVERITY_KEYS:
                row[f"plain_{s}_wer"] = m["severity"].get(s, float("nan"))
        else:
            row["plain_overall_wer"] = float("nan")
            for s in SEVERITY_KEYS:
                row[f"plain_{s}_wer"] = float("nan")


        b_v = _read_json(_setting_b_path(whisper, llm_tag, seed, b_root))
        if b_v is not None:
            m = _per_seed_metrics(b_v)
            row["B_vgate_overall_wer"] = m["overall_wer"]
            for s in SEVERITY_KEYS:
                row[f"B_vgate_{s}_wer"] = m["severity"].get(s, float("nan"))
            row["B_acoustic_fds_reference"] = b_v.get("acoustic_fds_reference", "")
        else:
            row["B_vgate_overall_wer"] = float("nan")
            for s in SEVERITY_KEYS:
                row[f"B_vgate_{s}_wer"] = float("nan")
            row["B_acoustic_fds_reference"] = ""


        corr_payload = _read_json(_setting_b_corr_path(whisper, llm_tag, seed, b_root))
        if corr_payload is not None:
            gate_rows.append(
                {
                    "whisper": whisper,
                    "llm": llm_tag,
                    "seed": seed,
                    "pearson_r": float(corr_payload.get("pearson_r", float("nan"))),
                    "spearman_rho": float(corr_payload.get("spearman_rho", float("nan"))),
                    "n_frames": int(corr_payload.get("n_frames_total", 0) or 0),
                    "n_test_utts_used": int(corr_payload.get("n_test_utts_used", 0) or 0),
                    "gate_norm_mean": float(corr_payload.get("gate_norm_mean", float("nan"))),
                    "fds_mean": float(corr_payload.get("fds_mean", float("nan"))),
                }
            )
        rows.append(row)
    return rows, gate_rows


def _aggregate_pair(
    rows: list[dict],
    gate_rows: list[dict],
    plain_fallback: dict | None,
) -> dict:

    if not rows:
        raise ValueError("no rows to aggregate")
    df = pd.DataFrame(rows)

    headline: dict = {}
    headline["whisper"] = rows[0]["whisper"]
    headline["llm"] = rows[0]["llm"]
    headline["n_seeds_paired"] = int(
        ((~df["A_vgate_overall_wer"].isna()) & (~df["B_vgate_overall_wer"].isna())).sum()
    )

    a_wer_mean, a_wer_std, _ = _mean_std_pct(df["A_vgate_overall_wer"])
    b_wer_mean, b_wer_std, _ = _mean_std_pct(df["B_vgate_overall_wer"])
    a_modsev_mean, a_modsev_std, _ = _mean_std_pct(df[f"A_vgate_{HIGH_HEADROOM_BIN}_wer"])
    b_modsev_mean, b_modsev_std, _ = _mean_std_pct(df[f"B_vgate_{HIGH_HEADROOM_BIN}_wer"])


    if df["plain_overall_wer"].notna().any():
        plain_overall_mean, plain_overall_std, _ = _mean_std_pct(df["plain_overall_wer"])
        plain_modsev_mean, plain_modsev_std, _ = _mean_std_pct(df[f"plain_{HIGH_HEADROOM_BIN}_wer"])
        plain_source = "paper_personalized_train"
    elif plain_fallback is not None:
        plain_overall_mean = plain_fallback.get("overall_wer_mean_pct", float("nan"))
        plain_overall_std = plain_fallback.get("overall_wer_std_pct", float("nan"))
        plain_sev = plain_fallback.get("severity", {}).get(HIGH_HEADROOM_BIN, {})
        plain_modsev_mean = plain_sev.get("wer_mean_pct", float("nan"))
        plain_modsev_std = plain_sev.get("wer_std_pct", float("nan"))
        plain_source = plain_fallback.get("source", "paper_reference")
    else:
        plain_overall_mean = float("nan")
        plain_overall_std = float("nan")
        plain_modsev_mean = float("nan")
        plain_modsev_std = float("nan")
        plain_source = "missing"

    gain_a_overall = (
        (plain_overall_mean - a_wer_mean) if not np.isnan(plain_overall_mean) else float("nan")
    )
    gain_b_overall = (
        (plain_overall_mean - b_wer_mean) if not np.isnan(plain_overall_mean) else float("nan")
    )
    gain_a_modsev = (
        (plain_modsev_mean - a_modsev_mean) if not np.isnan(plain_modsev_mean) else float("nan")
    )
    gain_b_modsev = (
        (plain_modsev_mean - b_modsev_mean) if not np.isnan(plain_modsev_mean) else float("nan")
    )

    delta_overall = b_wer_mean - a_wer_mean
    delta_modsev = b_modsev_mean - a_modsev_mean


    paired = df.dropna(subset=["A_vgate_overall_wer", "B_vgate_overall_wer"]).copy()
    paired_delta_overall_pp = float("nan")
    paired_delta_modsev_pp = float("nan")
    if not paired.empty:
        diffs = (paired["B_vgate_overall_wer"] - paired["A_vgate_overall_wer"]) * 100.0
        paired_delta_overall_pp = float(diffs.mean())
        diffs_m = (
            paired[f"B_vgate_{HIGH_HEADROOM_BIN}_wer"] - paired[f"A_vgate_{HIGH_HEADROOM_BIN}_wer"]
        ) * 100.0
        if diffs_m.notna().any():
            paired_delta_modsev_pp = float(diffs_m.mean(skipna=True))

    if gate_rows:
        gate_df = pd.DataFrame(gate_rows)
        gate_pearson_mean = float(gate_df["pearson_r"].mean(skipna=True))
        gate_pearson_std = float(gate_df["pearson_r"].std(skipna=True))
        gate_spearman_mean = float(gate_df["spearman_rho"].mean(skipna=True))
        gate_spearman_std = float(gate_df["spearman_rho"].std(skipna=True))
        gate_n_seeds = int(gate_df["pearson_r"].notna().sum())
    else:
        gate_pearson_mean = gate_pearson_std = float("nan")
        gate_spearman_mean = gate_spearman_std = float("nan")
        gate_n_seeds = 0

    bucket = _classify(delta_overall, delta_modsev, gain_a_overall, gain_b_overall)

    headline.update(
        {
            "A_overall_wer_mean_pct": a_wer_mean,
            "A_overall_wer_std_pct": a_wer_std,
            "B_overall_wer_mean_pct": b_wer_mean,
            "B_overall_wer_std_pct": b_wer_std,
            "delta_overall_pp": delta_overall,
            "paired_delta_overall_pp": paired_delta_overall_pp,
            "A_modsev_wer_mean_pct": a_modsev_mean,
            "A_modsev_wer_std_pct": a_modsev_std,
            "B_modsev_wer_mean_pct": b_modsev_mean,
            "B_modsev_wer_std_pct": b_modsev_std,
            "delta_modsev_pp": delta_modsev,
            "paired_delta_modsev_pp": paired_delta_modsev_pp,
            "plain_overall_wer_mean_pct": plain_overall_mean,
            "plain_overall_wer_std_pct": plain_overall_std,
            "plain_modsev_wer_mean_pct": plain_modsev_mean,
            "plain_modsev_wer_std_pct": plain_modsev_std,
            "plain_source": plain_source,
            "gain_A_overall_pp": gain_a_overall,
            "gain_B_overall_pp": gain_b_overall,
            "gain_A_modsev_pp": gain_a_modsev,
            "gain_B_modsev_pp": gain_b_modsev,
            "B_gate_fds_pearson_mean": gate_pearson_mean,
            "B_gate_fds_pearson_std": gate_pearson_std,
            "B_gate_fds_spearman_mean": gate_spearman_mean,
            "B_gate_fds_spearman_std": gate_spearman_std,
            "B_gate_fds_n_seeds": gate_n_seeds,
            "verdict": bucket,
        }
    )
    return headline


def _fds_distribution_rows(seeds: list[int], cache_root: Path) -> list[dict]:
    rows = []
    a_path = cache_root / "fds_cache" / "raw_frame_scores.npz"
    a_mean, a_std, a_n = _load_raw_score_dist(a_path)
    rows.append(
        {
            "scope": "all_control",
            "seed": "",
            "raw_score_mean": a_mean,
            "raw_score_std": a_std,
            "n_frames": a_n,
            "path": str(a_path),
        }
    )
    seed_means = []
    seed_stds = []
    for s in seeds:
        p = cache_root / "fds_cache_train_only" / f"seed{s}" / "raw_frame_scores.npz"
        m, sd, n = _load_raw_score_dist(p)
        rows.append(
            {
                "scope": f"train_only_seed{s}",
                "seed": s,
                "raw_score_mean": m,
                "raw_score_std": sd,
                "n_frames": n,
                "path": str(p),
            }
        )
        if not np.isnan(m):
            seed_means.append(m)
            seed_stds.append(sd)
    if seed_means:
        rows.append(
            {
                "scope": "train_only_seed_avg",
                "seed": "",
                "raw_score_mean": float(np.mean(seed_means)),
                "raw_score_std": float(np.mean(seed_stds)),
                "n_frames": -1,
                "path": "(mean across seeds)",
            }
        )
    return rows



def _fmt(val: float, digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:.{digits}f}"


def _render_markdown(
    headline_rows: list[dict],
    gate_rows: list[dict],
    fds_dist_rows: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# FDS reference split ablation: Train-only FDS (B) vs All-control FDS (A)\n")
    lines.append(
        "WER values are aggregated across the 10 paper seeds "
        "(`42,123,456,789,999,2024,2025,3407,777,888`). "
        "Setting A is read from `data/paper_personalized_train/...`. "
        "Setting B is read from `results/torgo/runs/fds_vgate_train_only_main/...` "
        "(this run's checkpoints).\n"
    )

    lines.append("## Headline comparison\n")
    lines.append(
        "| Whisper | LLM | n_seeds | A overall WER | B overall WER | ΔWER (B-A) | "
        "A mod.-sev. WER | B mod.-sev. WER | Δmod.-sev. (B-A) | "
        "gain A (Plain-VGate) | gain B (Plain-VGate) | verdict |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for h in headline_rows:
        lines.append(
            "| {whisper} | {llm} | {n} | {a_w} | {b_w} | {dw} | {a_m} | {b_m} | {dm} | "
            "{ga} | {gb} | {v} |".format(
                whisper=h["whisper"],
                llm=h["llm"],
                n=h["n_seeds_paired"],
                a_w=f"{_fmt(h['A_overall_wer_mean_pct'])} ± {_fmt(h['A_overall_wer_std_pct'])}",
                b_w=f"{_fmt(h['B_overall_wer_mean_pct'])} ± {_fmt(h['B_overall_wer_std_pct'])}",
                dw=_fmt(h["delta_overall_pp"]),
                a_m=f"{_fmt(h['A_modsev_wer_mean_pct'])} ± {_fmt(h['A_modsev_wer_std_pct'])}",
                b_m=f"{_fmt(h['B_modsev_wer_mean_pct'])} ± {_fmt(h['B_modsev_wer_std_pct'])}",
                dm=_fmt(h["delta_modsev_pp"]),
                ga=_fmt(h["gain_A_overall_pp"]),
                gb=_fmt(h["gain_B_overall_pp"]),
                v=h["verdict"],
            )
        )
    lines.append("")

    lines.append("## Gate-FDS correlation (Setting B only; A side checkpoints not preserved)\n")
    lines.append("| Whisper | LLM | n_seeds (B) | Pearson r | Spearman ρ |")
    lines.append("|---|---|---|---|---|")
    for h in headline_rows:
        lines.append(
            "| {w} | {l} | {n} | {pr} | {sp} |".format(
                w=h["whisper"],
                l=h["llm"],
                n=h["B_gate_fds_n_seeds"],
                pr=f"{_fmt(h['B_gate_fds_pearson_mean'], 3)} ± {_fmt(h['B_gate_fds_pearson_std'], 3)}",
                sp=f"{_fmt(h['B_gate_fds_spearman_mean'], 3)} ± {_fmt(h['B_gate_fds_spearman_std'], 3)}",
            )
        )
    lines.append(
        "\nA-side correlations are not reported because the All-control checkpoints "
        "for these cells were not preserved at paper time (see plan / future work).\n"
    )

    lines.append("## FDS reference distribution (raw frame scores)\n")
    lines.append("| Scope | Seed | mean | std | n_frames |")
    lines.append("|---|---|---|---|---|")
    for r in fds_dist_rows:
        nf = r["n_frames"]
        nf_str = "n/a" if nf == 0 else ("(avg)" if nf == -1 else f"{nf:,}")
        lines.append(
            f"| {r['scope']} | {r['seed']} | {_fmt(r['raw_score_mean'], 3)} | "
            f"{_fmt(r['raw_score_std'], 3)} | {nf_str} |"
        )
    lines.append("")

    lines.append("## Per-seed pearson/spearman raw values (B side)\n")
    lines.append("| Whisper | LLM | Seed | Pearson r | Spearman ρ | n_frames |")
    lines.append("|---|---|---|---|---|---|")
    for g in gate_rows:
        lines.append(
            f"| {g['whisper']} | {g['llm']} | {g['seed']} | "
            f"{_fmt(g['pearson_r'], 3)} | {_fmt(g['spearman_rho'], 3)} | {g['n_frames']:,} |"
        )

    lines.append("\n## Notes\n")
    lines.append(
        "- WER columns are percent. ΔWER (B−A) is in absolute percentage points (pp).\n"
        "- `gain X (Plain-VGate)` = `Plain WER − VGate WER` (positive = VGate beats Plain).\n"
        "- `verdict` follows the success-criteria buckets in the ablation plan.\n"
    )
    return "\n".join(lines)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Train-only FDS (B) vs All-control FDS (A) results."
    )
    parser.add_argument(
        "--setting-a-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_personalized_train",
        help="Setting A WER source.",
    )
    parser.add_argument(
        "--plain-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_personalized_train",
        help="Plain baseline per-seed source; falls back to paper_table_reference.csv.",
    )
    parser.add_argument(
        "--paper-reference-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "paper_reference" / "paper_table_reference.csv",
        help="Aggregated paper reference (used for plain WER fallback).",
    )
    parser.add_argument(
        "--train-only-results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "torgo" / "runs" / "fds_vgate_train_only_main",
        help="Setting B WER + gate-FDS source.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "cache" / "whisper_embeddings" / "torgo_whisper_largev2",
        help="Whisper-large-v2 cache root holding fds_cache/ and fds_cache_train_only/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination dir; defaults to <train-only-results-root>/_summary.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=",".join(f"{w}:{l}" for w, l in DEFAULT_PAIRS),
        help="Comma-separated <whisper>:<llm_tag> pairs.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pair_specs = []
    for spec in args.pairs.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(f"Bad --pairs entry: {spec}")
        w, l = spec.split(":", 1)
        pair_specs.append((w, l))

    out_dir = args.out_dir or (args.train_only_results_root / "_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_ref_df = _load_paper_reference_csv(args.paper_reference_csv)

    all_per_seed_rows: list[dict] = []
    all_gate_rows: list[dict] = []
    headline_rows: list[dict] = []
    for whisper, llm_tag in pair_specs:
        per_seed, gate_rows = _per_seed_collect(
            whisper, llm_tag, seeds,
            args.setting_a_root, args.train_only_results_root,
            args.plain_root,
        )
        all_per_seed_rows.extend(per_seed)
        all_gate_rows.extend(gate_rows)
        plain_fb = _plain_aggregate_from_reference(paper_ref_df, whisper, llm_tag)
        headline = _aggregate_pair(per_seed, gate_rows, plain_fb)
        headline_rows.append(headline)

    fds_rows = _fds_distribution_rows(seeds, args.cache_root)

    per_seed_csv = out_dir / "compare_fds_ref_ablation_per_seed.csv"
    headline_csv = out_dir / "compare_fds_ref_ablation.csv"
    gate_csv = out_dir / "compare_fds_ref_ablation_gate_fds.csv"
    fds_csv = out_dir / "compare_fds_ref_ablation_fds_dist.csv"
    md_path = out_dir / "compare_fds_ref_ablation.md"

    pd.DataFrame(all_per_seed_rows).to_csv(per_seed_csv, index=False)
    pd.DataFrame(headline_rows).to_csv(headline_csv, index=False)
    pd.DataFrame(all_gate_rows).to_csv(gate_csv, index=False)
    pd.DataFrame(fds_rows).to_csv(fds_csv, index=False)
    md_path.write_text(_render_markdown(headline_rows, all_gate_rows, fds_rows), encoding="utf-8")

    print(f"[done] wrote per-cell headline -> {headline_csv}")
    print(f"[done] wrote per-seed rows     -> {per_seed_csv}")
    print(f"[done] wrote gate-FDS table    -> {gate_csv}")
    print(f"[done] wrote FDS dist table    -> {fds_csv}")
    print(f"[done] wrote markdown report   -> {md_path}")
    print()
    for h in headline_rows:
        print(
            f"  {h['whisper']:>7s} / {h['llm']:>9s}  "
            f"A={_fmt(h['A_overall_wer_mean_pct'])}  B={_fmt(h['B_overall_wer_mean_pct'])}  "
            f"Δ={_fmt(h['delta_overall_pp'])}pp  -> {h['verdict']}"
        )


if __name__ == "__main__":
    main()
