from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

SEVERITY_KEYS = ("Severe", "Moderately-severe", "Moderate", "Mild", "control")

_ORIG_TO_FSAC_MODE = {
    "dte_off": "fds_off",
    "vgate": "vgate",
    "fds_off": "fds_off",
    "plain": "plain",
}

_RUN_DIR_RE = re.compile(r"^whispering_l(?:lama|lm)_(?P<mode>.+)_s(?P<seed>\d+)$")
_FOLD_DIR_RE = re.compile(r"^fold_(?P<fold>[A-Za-z0-9]+)$")
_K_DIR_RE = re.compile(r"^k(?P<k>[A-Za-z0-9]+)$")
_SEED_DIR_RE = re.compile(r"^seed(?P<seed>\d+)$")


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(var)


def _to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def _fmt(value: float | None, places: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{places}f}"


def _load_cell(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for sev in SEVERITY_KEYS:
        bin_data = (payload.get("severity_breakdown") or {}).get(sev)
        if not bin_data:
            continue
        out[sev] = {
            "wer": _to_percent(bin_data.get("wer")),
            "cer": _to_percent(bin_data.get("cer")),
            "n":   bin_data.get("n"),
        }
    out["overall"] = {
        "wer": _to_percent(payload.get("overall_wer")),
        "cer": _to_percent(payload.get("overall_cer")),
        "n":   payload.get("n_samples"),
    }
    return out


def _iter_loso_cells(scan_root: Path):











    bad = re.compile(r"\.bak")
    if not scan_root.is_dir():
        return
    for whisper_dir in sorted(p for p in scan_root.iterdir() if p.is_dir()):
        whisper = whisper_dir.name
        if bad.search(whisper) or whisper.startswith("_"):
            continue
        for llm_dir in sorted(p for p in whisper_dir.iterdir() if p.is_dir()):
            llm = llm_dir.name
            if bad.search(llm):
                continue
            for fold_dir in sorted(p for p in llm_dir.iterdir() if p.is_dir()):
                m_fold = _FOLD_DIR_RE.match(fold_dir.name)
                if not m_fold:
                    continue
                fold = m_fold.group("fold")
                for k_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
                    m_k = _K_DIR_RE.match(k_dir.name)
                    if not m_k:
                        continue
                    k = m_k.group("k")
                    for seed_dir in sorted(p for p in k_dir.iterdir() if p.is_dir()):
                        m_seed = _SEED_DIR_RE.match(seed_dir.name)
                        if not m_seed:
                            continue
                        seed = m_seed.group("seed")
                        for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                            m_run = _RUN_DIR_RE.match(run_dir.name)
                            if not m_run:
                                continue
                            raw_mode = m_run.group("mode")
                            mode = _ORIG_TO_FSAC_MODE.get(raw_mode, raw_mode)
                            json_path = run_dir / "inference_results.json"
                            yield whisper, llm, fold, k, seed, mode, json_path


def _aggregate(cell_iter, group_keys):









    DIM_NAMES = ("whisper", "llm_tag", "fold", "k", "seed", "mode")
    bins: dict[tuple, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"wer": [], "cer": [], "n": [], "n_cells": 0})
    )

    for cell in cell_iter:
        whisper, llm, fold, k, seed, mode, json_path = cell
        record = _load_cell(json_path)
        if not record:
            continue
        all_vals = {"whisper": whisper, "llm_tag": llm, "fold": fold,
                    "k": k, "seed": seed, "mode": mode}
        key = tuple(all_vals[d] for d in group_keys)
        for sev, metrics in record.items():
            slot = bins[key][sev]
            slot["wer"].append(metrics["wer"])
            slot["cer"].append(metrics["cer"])
            if metrics["n"] is not None:
                slot["n"].append(metrics["n"])
            slot["n_cells"] += 1

    rows: list[dict] = []
    for key, by_sev in sorted(bins.items()):
        base = dict(zip(group_keys, key))
        for sev in ("overall", *SEVERITY_KEYS):
            if sev not in by_sev:
                continue
            slot = by_sev[sev]
            wer_mean, wer_std = _mean_std(slot["wer"])
            cer_mean, cer_std = _mean_std(slot["cer"])
            n_total = sum(slot["n"]) if slot["n"] else None
            row = {**base, "severity": sev,
                   "wer_mean_%": wer_mean, "wer_std_%": wer_std,
                   "cer_mean_%": cer_mean, "cer_std_%": cer_std,
                   "n_total_utterances": n_total, "n_cells": slot["n_cells"]}
            rows.append(row)
    return rows


def _write_csv(rows: list[dict], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt_cols = {"wer_mean_%", "wer_std_%", "cer_mean_%", "cer_std_%"}
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for col in fieldnames:
                val = row.get(col)
                if col in fmt_cols:
                    out[col] = _fmt(val, places=2)
                elif val is None:
                    out[col] = ""
                else:
                    out[col] = val
            writer.writerow(out)


def _write_md_headline(rows: list[dict], out_path: Path) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_sev: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sev[row["severity"]].append(row)

    lines = ["# LOSO + k-shot WER/CER (mean +/- std across folds x k x seeds)\n"]
    for sev in ("overall", *SEVERITY_KEYS):
        if sev not in by_sev:
            continue
        lines.append(f"\n## {sev}\n")
        lines.append("| Whisper | LLM | Mode | WER% (mean +/- std) | CER% (mean +/- std) | cells |")
        lines.append("|---|---|---|---|---|---|")
        for row in sorted(by_sev[sev], key=lambda r: (r["whisper"], r["llm_tag"], r["mode"])):
            wer = _fmt(row["wer_mean_%"]) + " +/- " + _fmt(row["wer_std_%"])
            cer = _fmt(row["cer_mean_%"]) + " +/- " + _fmt(row["cer_std_%"])
            lines.append(
                f"| {row['whisper']} | {row['llm_tag']} | {row['mode']} | {wer} | {cer} | {row['n_cells']} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md_fold(rows: list[dict], out_path: Path) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_sev: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sev[row["severity"]].append(row)

    lines = ["# LOSO + k-shot WER/CER per fold (mean +/- std across seeds)\n"]
    for sev in ("overall", *SEVERITY_KEYS):
        if sev not in by_sev:
            continue
        lines.append(f"\n## {sev}\n")
        lines.append("| Whisper | LLM | Fold | K | Mode | WER% (mean +/- std) | CER% (mean +/- std) | cells |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in sorted(by_sev[sev], key=lambda r: (r["whisper"], r["llm_tag"], r["fold"], r["k"], r["mode"])):
            wer = _fmt(row["wer_mean_%"]) + " +/- " + _fmt(row["wer_std_%"])
            cer = _fmt(row["cer_mean_%"]) + " +/- " + _fmt(row["cer_std_%"])
            lines.append(
                f"| {row['whisper']} | {row['llm_tag']} | {row['fold']} | {row['k']} "
                f"| {row['mode']} | {wer} | {cer} | {row['n_cells']} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-root", type=Path, required=True,
                        help="root of the LOSO results tree "
                             "(<scan_root>/<whisper>/<llm>/fold_*/k*/seed*/whispering_l*_<mode>_s*).")
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="path for headline (mode x whisper x llm) CSV.")
    parser.add_argument("--out-md", type=Path, required=True,
                        help="path for headline markdown table.")
    parser.add_argument("--fold-csv", type=Path,
                        help="path for per-fold breakdown CSV (optional).")
    parser.add_argument("--fold-md", type=Path,
                        help="path for per-fold breakdown markdown (optional).")
    args = parser.parse_args()

    if not args.scan_root.is_dir():
        raise SystemExit(f"scan-root is not a directory: {args.scan_root}")

    headline_rows = _aggregate(_iter_loso_cells(args.scan_root),
                               group_keys=("whisper", "llm_tag", "mode"))
    if not headline_rows:
        raise SystemExit(f"no inference_results.json found under {args.scan_root}")

    headline_cols = ["whisper", "llm_tag", "mode", "severity",
                     "wer_mean_%", "wer_std_%", "cer_mean_%", "cer_std_%",
                     "n_total_utterances", "n_cells"]
    _write_csv(headline_rows, args.out_csv, headline_cols)
    _write_md_headline(headline_rows, args.out_md)
    print(f"wrote {args.out_csv}  ({len(headline_rows)} rows)")
    print(f"wrote {args.out_md}")

    if args.fold_csv or args.fold_md:
        fold_rows = _aggregate(_iter_loso_cells(args.scan_root),
                               group_keys=("whisper", "llm_tag", "fold", "k", "mode"))
        fold_cols = ["whisper", "llm_tag", "fold", "k", "mode", "severity",
                     "wer_mean_%", "wer_std_%", "cer_mean_%", "cer_std_%",
                     "n_total_utterances", "n_cells"]
        if args.fold_csv:
            _write_csv(fold_rows, args.fold_csv, fold_cols)
            print(f"wrote {args.fold_csv}  ({len(fold_rows)} rows)")
        if args.fold_md:
            _write_md_fold(fold_rows, args.fold_md)
            print(f"wrote {args.fold_md}")


if __name__ == "__main__":
    main()
