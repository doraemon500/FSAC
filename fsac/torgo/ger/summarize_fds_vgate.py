from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


WHISPERS = ("tiny", "base", "small", "medium", "largev2")
LLMS = ("qwen3_4b", "qwen3_8b")
MODES = {
    "plain": "whispering_llm_plain_s{seed}",
    "fds_off": "whispering_llm_fds_off_s{seed}",
    "vgate": "whispering_llm_vgate_s{seed}",
}


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect(result_root: Path):
    rows = []
    severity_rows = []
    for whisper_dir in sorted(result_root.iterdir() if result_root.exists() else []):
        if not whisper_dir.is_dir() or whisper_dir.name.startswith("_"):
            continue
        whisper = whisper_dir.name
        for llm_dir in sorted(whisper_dir.iterdir()):
            if not llm_dir.is_dir():
                continue
            llm = llm_dir.name
            for seed_dir in sorted(llm_dir.glob("seed*")):
                if not seed_dir.is_dir():
                    continue
                seed = seed_dir.name.removeprefix("seed")
                for mode, dirname in MODES.items():
                    result = _read_json(seed_dir / dirname.format(seed=seed) / "inference_results.json")
                    if result is None:
                        continue
                    rows.append({
                        "whisper": whisper,
                        "llm": llm,
                        "seed": seed,
                        "mode": mode,
                        "wer": result.get("overall_wer"),
                        "cer": result.get("overall_cer"),
                        "n_samples": result.get("n_samples"),
                        "checkpoint_epoch": result.get("checkpoint_epoch"),
                        "checkpoint_val_loss": result.get("checkpoint_val_loss"),
                    })
                    for severity, metrics in result.get("severity_breakdown", {}).items():
                        severity_rows.append({
                            "whisper": whisper,
                            "llm": llm,
                            "seed": seed,
                            "mode": mode,
                            "severity": severity,
                            "wer": metrics.get("wer"),
                            "cer": metrics.get("cer"),
                            "n": metrics.get("n"),
                        })
    return rows, severity_rows


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _delta_rows(severity_rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in severity_rows:
        key = (row["whisper"], row["llm"], row["seed"], row["severity"])
        by_key.setdefault(key, {})[row["mode"]] = row
    out = []
    for (whisper, llm, seed, severity), modes in sorted(by_key.items()):
        off = modes.get("fds_off")
        vg = modes.get("vgate")
        if not off or not vg:
            continue
        off_wer = float(off["wer"])
        vg_wer = float(vg["wer"])
        out.append({
            "whisper": whisper,
            "llm": llm,
            "seed": seed,
            "severity": severity,
            "fds_off_wer": off_wer,
            "vgate_wer": vg_wer,
            "vgate_delta_vs_fds_off": vg_wer - off_wer,
            "n": vg.get("n"),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Summarize TORGO FSAC results.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.result_root / "_summary")
    rows, severity_rows = _collect(args.result_root)
    _write_csv(out_dir / "summary_by_run.csv", rows)
    _write_csv(out_dir / "severity_results.csv", severity_rows)
    _write_csv(out_dir / "severity_delta_vs_fds_off.csv", _delta_rows(severity_rows))
    print(f"Wrote summary CSVs to {out_dir}")


if __name__ == "__main__":
    main()
