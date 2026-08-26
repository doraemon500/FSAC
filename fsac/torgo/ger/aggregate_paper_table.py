from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

SEVERITY_KEYS = ("Severe", "Moderately-severe", "Moderate", "Mild", "control")

_ORIG_TO_FSAC_MODE = {
    "dte_off": "fds_off",
    "vgate": "vgate",
    "fds_off": "fds_off",
    "plain": "plain",
}



_RUN_DIR_RE = re.compile(r"^whispering_l(?:lama|lm)_(?P<mode>.+)_s(?P<seed>\d+)$")


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
    payload = json.loads(json_path.read_text(encoding="utf-8"))
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


def _iter_manifest_cells(manifest_path: Path):

    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_dir = Path(row["run_dir"])
            json_path = run_dir / f"whispering_llm{row['fsac_suffix']}" / "inference_results.json"
            yield row["whisper"], row["llm_tag"], row["mode"], row["seed"], json_path


def _iter_scan_root_cells(scan_root: Path):











    bad = re.compile(r"\.bak")
    for whisper_dir in sorted(p for p in scan_root.iterdir() if p.is_dir()):
        whisper = whisper_dir.name
        if bad.search(whisper):
            continue
        for llm_dir in sorted(p for p in whisper_dir.iterdir() if p.is_dir()):
            llm_tag = llm_dir.name
            if bad.search(llm_tag):
                continue
            for seed_dir in sorted(p for p in llm_dir.iterdir() if p.is_dir() and p.name.startswith("seed")):
                seed = seed_dir.name[len("seed"):]
                for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                    m = _RUN_DIR_RE.match(run_dir.name)
                    if not m:
                        continue
                    if m.group("seed") != seed:
                        continue
                    json_path = run_dir / "inference_results.json"
                    if not json_path.exists():
                        continue
                    raw_mode = m.group("mode")
                    mode = _ORIG_TO_FSAC_MODE.get(raw_mode, raw_mode)
                    yield whisper, llm_tag, mode, seed, json_path


def _aggregate(cell_iter) -> list[dict]:


    groups: dict[tuple[str, str, str], list[dict]] = {}
    for whisper, llm_tag, mode, seed, json_path in cell_iter:
        cell = _load_cell(json_path)
        if not cell:
            print(f"WARN: missing inference_results.json -> {json_path}")
            continue
        key = (whisper, llm_tag, mode)
        groups.setdefault(key, []).append({"seed": seed, "cell": cell})




    seed_counts = {key: len(cells) for key, cells in groups.items()}
    if seed_counts:
        modal = max(set(seed_counts.values()), key=lambda v: list(seed_counts.values()).count(v))
        for key, n in sorted(seed_counts.items()):
            if n != modal:
                print(
                    f"WARN: cell {'/'.join(key)} has {n} seeds (modal={modal})"
                )

    aggregated: list[dict] = []
    for (whisper, llm_tag, mode), cells in groups.items():
        seeds_used = sorted(c["seed"] for c in cells)
        for sev in (*SEVERITY_KEYS, "overall"):
            wers = [c["cell"].get(sev, {}).get("wer") for c in cells]
            cers = [c["cell"].get(sev, {}).get("cer") for c in cells]
            ns   = [c["cell"].get(sev, {}).get("n")   for c in cells if c["cell"].get(sev)]
            if all(v is None for v in wers) and all(v is None for v in cers):
                continue
            wer_mean, wer_std = _mean_std(wers)
            cer_mean, cer_std = _mean_std(cers)
            aggregated.append({
                "whisper":   whisper,
                "llm_tag":   llm_tag,
                "mode":      mode,
                "severity":  sev,
                "n_seeds":   len(cells),
                "seeds":     ",".join(seeds_used),
                "n_utts":    ns[0] if ns else "",
                "wer_mean":  wer_mean,
                "wer_std":   wer_std,
                "cer_mean":  cer_mean,
                "cer_std":   cer_std,
            })
    return aggregated


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "whisper", "llm_tag", "mode", "severity",
        "n_seeds", "seeds", "n_utts",
        "wer_mean", "wer_std", "cer_mean", "cer_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "wer_mean": _fmt(row["wer_mean"]),
                "wer_std":  _fmt(row["wer_std"]),
                "cer_mean": _fmt(row["cer_mean"]),
                "cer_std":  _fmt(row["cer_std"]),
            })


def _write_md(rows: list[dict], path: Path) -> None:
    by_severity: dict[str, list[dict]] = {}
    for row in rows:
        by_severity.setdefault(row["severity"], []).append(row)

    severity_order = (*SEVERITY_KEYS, "overall")

    lines: list[str] = ["# Paper table reproduction (TORGO)", ""]
    for sev in severity_order:
        if sev not in by_severity:
            continue
        lines.append(f"## Severity: {sev}")
        lines.append("")
        lines.append(
            "| Whisper | LLM | Mode | n_seeds | n_utts | "
            "WER mean | WER std | CER mean | CER std |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|"
        )
        sub = sorted(by_severity[sev], key=lambda r: (r["whisper"], r["llm_tag"], r["mode"]))
        for r in sub:
            lines.append(
                f"| {r['whisper']} | {r['llm_tag']} | {r['mode']} | "
                f"{r['n_seeds']} | {r['n_utts']} | "
                f"{_fmt(r['wer_mean'])} | {_fmt(r['wer_std'])} | "
                f"{_fmt(r['cer_mean'])} | {_fmt(r['cer_std'])} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate FSAC inference_results.json into the paper table."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--manifest",
        type=Path,
        help="Manifest written by a checkpoint reproduction script.",
    )
    src.add_argument(
        "--scan-root",
        type=Path,
        help="Walk an existing result tree (paper checkpoints layout) and "
        "aggregate every inference_results.json found.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        type=Path,
        help="Destination CSV for the aggregated table.",
    )
    parser.add_argument(
        "--out-md",
        required=True,
        type=Path,
        help="Destination markdown copy of the aggregated table.",
    )
    args = parser.parse_args()

    if args.manifest is not None:
        cell_iter = _iter_manifest_cells(args.manifest)
    else:
        if not args.scan_root.is_dir():
            raise SystemExit(f"scan-root is not a directory: {args.scan_root}")
        cell_iter = _iter_scan_root_cells(args.scan_root)

    rows = _aggregate(cell_iter)
    if not rows:
        raise SystemExit("aggregate_paper_table: no inference results found")
    _write_csv(rows, args.out_csv)
    _write_md(rows, args.out_md)

    n_cells = len({(r["whisper"], r["llm_tag"], r["mode"]) for r in rows})
    print(f"aggregated {n_cells} cells -> {args.out_csv}")
    print(f"markdown copy -> {args.out_md}")


if __name__ == "__main__":
    main()
