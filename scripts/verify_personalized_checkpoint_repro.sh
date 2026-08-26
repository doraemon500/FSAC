#!/bin/bash
# Checkpoint-only reproduction verifier for the TORGO personalized grid.
#
# This runs FSAC inference only. It does not train and does not mutate the
# original checkpoint tree. Generated outputs are written under OUT_ROOT, then
# compared against the original predictions/metrics with schema-aware diffs.
#
# Default grid:
#   5 Whispers x 2 LLMs x 10 seeds x {plain,fds_off,vgate} = 300 cells
#
# Example:
#   GPUS=0,1,2,3,4 bash scripts/verify_personalized_checkpoint_repro.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/lib/gpu_queue.sh

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: python/python3 not found in PATH" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${ROOT}/results/torgo/runs/personalized_v1_promptablate_v3}"
FDS_EMBEDDINGS_ROOT="${FDS_EMBEDDINGS_ROOT:-${ROOT}/cache/fds}"
FSAC_SPLITS_DIR="${FSAC_SPLITS_DIR:-${ROOT}/data/splits/torgo/personalized_v1_clean_no_bad_utt_v3}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/torgo/runs/repro_personalized_checkpoint_${TS}}"
GPUS="${GPUS:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

WHISPERS="${WHISPERS:-tiny,base,small,medium,large-v2}"
LLM_TAGS="${LLM_TAGS:-qwen3_4b,qwen3_8b}"
MODES="${MODES:-plain,fds_off,vgate}"
SEEDS="${SEEDS:-42,123,456,789,999,2024,2025,3407,777,888}"

export PYTHON_BIN
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPU_QUEUE_LOG_DIR="${OUT_ROOT}/_logs"
export GPU_QUEUE_FAIL_FAST="${GPU_QUEUE_FAIL_FAST:-1}"

MANIFEST="${OUT_ROOT}/_manifest.csv"
DIFF_REPORT="${OUT_ROOT}/_diff_report.csv"
DIFF_SUMMARY="${OUT_ROOT}/_diff_summary.txt"
FSAC_CACHE_DIR="${OUT_ROOT}/_cache"
ACOUSTIC_CACHE="${FSAC_CACHE_DIR}/whisper_embeddings/torgo_whisper_largev2"
TABLE_CSV="${OUT_ROOT}/paper_table.csv"
TABLE_MD="${OUT_ROOT}/paper_table.md"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
require_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 1; }; }
require_dir()  { [[ -d "$1" ]] || { echo "ERROR: missing directory: $1" >&2; exit 1; }; }

gpu_preflight() {
    log "GPU preflight"
    local smi_ok=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        local smi_out
        if smi_out="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>&1)"; then
            echo "${smi_out}"
            smi_ok=1
        else
            log "WARN: nvidia-smi failed: ${smi_out}"
        fi
    else
        log "WARN: nvidia-smi not found; inference will rely on torch CUDA availability"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    if [[ "${smi_ok}" -ne 1 ]]; then
        echo "ERROR: GPU preflight failed; refusing to run checkpoint inference on CPU" >&2
        exit 1
    fi
    "${PYTHON_BIN}" - <<'PYEOF'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: failed to import torch: {exc}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    raise SystemExit("ERROR: torch CUDA is not available; refusing to run checkpoint inference on CPU")
PYEOF
}

stage_acoustic_cache() {
    local src="${FDS_EMBEDDINGS_ROOT}/torgo_whisper_largev2"
    require_dir "${src}"
    require_dir "${src}/dte_cache"
    require_file "${src}/dte_cache/raw_frame_scores.npz"
    require_file "${src}/dte_cache/zscore_stats.json"

    mkdir -p "${ACOUSTIC_CACHE}/fds_cache"
    local n_npz=0
    for npz in "${src}"/*.npz; do
        [[ -f "${npz}" ]] || continue
        ln -sfn "${npz}" "${ACOUSTIC_CACHE}/$(basename "${npz}")"
        n_npz=$((n_npz + 1))
    done
    if [[ "${n_npz}" -eq 0 ]]; then
        echo "ERROR: no speaker .npz files in ${src}" >&2
        exit 1
    fi
    cp -f "${src}/dte_cache/raw_frame_scores.npz" "${ACOUSTIC_CACHE}/fds_cache/raw_frame_scores.npz"
    cp -f "${src}/dte_cache/zscore_stats.json" "${ACOUSTIC_CACHE}/fds_cache/zscore_stats.json"
    log "staged acoustic cache: ${ACOUSTIC_CACHE} (${n_npz} speakers)"
}

write_manifest() {
    mkdir -p "${OUT_ROOT}"
    "${PYTHON_BIN}" - "${CHECKPOINTS_ROOT}" "${FSAC_SPLITS_DIR}" "${OUT_ROOT}" \
        "${WHISPERS}" "${LLM_TAGS}" "${MODES}" "${SEEDS}" "${MANIFEST}" <<'PYEOF'
import csv
import sys
from pathlib import Path

ckpt_root = Path(sys.argv[1])
splits_dir = Path(sys.argv[2])
out_root = Path(sys.argv[3])
whispers = [s.strip() for s in sys.argv[4].split(",") if s.strip()]
llm_tags = [s.strip() for s in sys.argv[5].split(",") if s.strip()]
modes = [s.strip() for s in sys.argv[6].split(",") if s.strip()]
seeds = [s.strip() for s in sys.argv[7].split(",") if s.strip()]
manifest = Path(sys.argv[8])

whisper_dirs = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v2": "largev2",
    "largev2": "largev2",
}
llm_models = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}
mode_map = {
    "plain":   {"orig_mode": "plain",   "orig_suffix": "_plain",   "fsac_suffix": "_plain"},
    "fds_off": {"orig_mode": "dte_off", "orig_suffix": "_dte_off", "fsac_suffix": "_fds_off"},
    "vgate":   {"orig_mode": "vgate",   "orig_suffix": "_vgate",   "fsac_suffix": "_vgate"},
}

unknown_w = sorted(set(whispers) - set(whisper_dirs))
unknown_l = sorted(set(llm_tags) - set(llm_models))
unknown_m = sorted(set(modes) - set(mode_map))
if unknown_w:
    raise SystemExit(f"unknown WHISPERS: {unknown_w}")
if unknown_l:
    raise SystemExit(f"unknown LLM_TAGS: {unknown_l}")
if unknown_m:
    raise SystemExit(f"unknown MODES: {unknown_m}")

rows = []
for w in whispers:
    wd = whisper_dirs[w]
    for lt in llm_tags:
        for seed in seeds:
            seed_dir = ckpt_root / wd / lt / f"seed{seed}"
            for mode in modes:
                m = mode_map[mode]
                orig_run_dir = seed_dir / f"whispering_llama{m['orig_suffix']}_s{seed}"
                ckpt_path = orig_run_dir / "best_checkpoint.pt"
                prompts_csv = seed_dir / "ger_prompts.csv"
                frame_dte = seed_dir / "whisper_dte" / "frame_dte_scores.csv"
                split_file = splits_dir / f"seed{seed}.json"
                orig_json = orig_run_dir / "inference_results.json"
                orig_pred = seed_dir / f"whispering_llama{m['orig_suffix']}_s{seed}_predictions.csv"
                run_dir = out_root / wd / lt / f"seed{seed}" / mode
                missing = []
                for label, path in (
                    ("ckpt", ckpt_path),
                    ("prompts", prompts_csv),
                    ("frame_dte", frame_dte),
                    ("split", split_file),
                    ("orig_json", orig_json),
                    ("orig_pred", orig_pred),
                ):
                    if not path.exists():
                        missing.append(f"{label}={path}")
                if missing:
                    raise SystemExit(
                        f"missing inputs for {wd}/{lt}/seed{seed}/{mode}:\n  "
                        + "\n  ".join(missing)
                    )
                fsac_suffix = f"{m['fsac_suffix']}_s{seed}"
                cell = f"{wd}/{lt}/seed{seed}/{mode}"
                rows.append({
                    "cell": cell,
                    "whisper": wd,
                    "llm_tag": lt,
                    "llm_model": llm_models[lt],
                    "seed": seed,
                    "mode": mode,
                    "orig_mode": m["orig_mode"],
                    "fsac_suffix": fsac_suffix,
                    "ckpt": str(ckpt_path),
                    "prompts_csv": str(prompts_csv),
                    "frame_dte": str(frame_dte),
                    "split_file": str(split_file),
                    "split_tag": f"personalized_v1_clean_no_bad_utt_v3/seed{seed}",
                    "run_dir": str(run_dir),
                    "orig_json": str(orig_json),
                    "orig_pred": str(orig_pred),
                })

expected = len(whispers) * len(llm_tags) * len(seeds) * len(modes)
if len(rows) != expected:
    raise SystemExit(f"manifest cell count mismatch: got {len(rows)} expected {expected}")

manifest.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "cell", "whisper", "llm_tag", "llm_model", "seed", "mode", "orig_mode",
    "fsac_suffix", "ckpt", "prompts_csv", "frame_dte", "split_file",
    "split_tag", "run_dir", "orig_json", "orig_pred",
]
with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote manifest ({len(rows)} cells): {manifest}")
PYEOF
}

stage_run_inputs() {
    "${PYTHON_BIN}" - "${MANIFEST}" <<'PYEOF'
import csv
import sys
from pathlib import Path

rename = {
    "frame_dte_mean": "frame_fds_mean",
    "frame_dte_max": "frame_fds_max",
    "frame_dte_std": "frame_fds_std",
}

with Path(sys.argv[1]).open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "whisper_fds").mkdir(parents=True, exist_ok=True)

    prompts_link = run_dir / "ger_prompts.csv"
    if prompts_link.exists() or prompts_link.is_symlink():
        prompts_link.unlink()
    prompts_link.symlink_to(Path(row["prompts_csv"]).resolve())

    src = Path(row["frame_dte"])
    dst = run_dir / "whisper_fds" / "frame_fds_scores.csv"
    with src.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = [rename.get(name, name) for name in (reader.fieldnames or [])]
        records = list(reader)
    with dst.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rec in records:
            writer.writerow({rename.get(k, k): v for k, v in rec.items()})

print(f"staged inputs for {len(rows)} cells")
PYEOF
}

build_infer_command() {
    local llm_model="$1" run_dir="$2" split_file="$3" split_tag="$4" seed="$5" \
          mode="$6" fsac_suffix="$7" ckpt="$8" label="$9"
    local infer_mode_flag
    case "${mode}" in
        plain)   infer_mode_flag="--fds-mode plain" ;;
        fds_off) infer_mode_flag="--fds-mode fds_off" ;;
        vgate)   infer_mode_flag="--fds-mode vgate --real-frame-fds" ;;
        *) echo "Unknown mode: ${mode}" >&2; exit 1 ;;
    esac

    cat <<COMMAND
set -euo pipefail
cd "${ROOT}"
export FSAC_DATA_DIR="${ROOT}/data/torgo/raw"
export FSAC_CACHE_DIR="${FSAC_CACHE_DIR}"
export FSAC_RESULTS_DIR="${OUT_ROOT}"
export TORGO_RESULT_DIR="${run_dir}"
export TORGO_SPLIT_FILE="${split_file}"
export TORGO_SPLIT_TAG="${split_tag}"
export GER_ABLATION_SUFFIX="${fsac_suffix}"
export WHISPER_MODEL="openai/whisper-large-v2"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
echo "[\$(date '+%H:%M:%S')] [job=${label}] START checkpoint-only infer gpu=\${CUDA_VISIBLE_DEVICES:-?}"
"${PYTHON_BIN}" -m fsac.torgo.ger.infer_whispering_llm \\
  --model "${llm_model}" \\
  --checkpoint "${ckpt}" \\
  --prompts-csv "${run_dir}/ger_prompts.csv" \\
  ${infer_mode_flag} --fds-variant true \\
  --gpu 0 --seed "${seed}" \\
  --max-new-tokens 128 \\
  --risk-scale zscore --gate-arch mlp --gate-init 1.0 \\
  --attn-risk-mode v_only
COMMAND
}

run_jobs() {
    gpu_queue_init "${GPUS}"
    local queued=0 skipped=0
    while IFS=, read -r cell whisper llm_tag llm_model seed mode orig_mode fsac_suffix \
        ckpt prompts_csv frame_dte split_file split_tag run_dir orig_json orig_pred; do
        [[ "${cell}" == "cell" ]] && continue
        orig_pred="${orig_pred%$'\r'}"
        local out_dir="${run_dir}/whispering_llm${fsac_suffix}"
        local out_json="${out_dir}/inference_results.json"
        local out_pred="${out_dir}/predictions.csv"
        if [[ -f "${out_json}" && -f "${out_pred}" && "${FORCE}" != "1" ]]; then
            log "SKIP existing: ${cell}"
            skipped=$((skipped + 1))
            continue
        fi
        local cmd
        cmd="$(build_infer_command "${llm_model}" "${run_dir}" "${split_file}" "${split_tag}" \
              "${seed}" "${mode}" "${fsac_suffix}" "${ckpt}" "${cell}")"
        gpu_queue_submit "${cmd}" "${cell}"
        queued=$((queued + 1))
    done < "${MANIFEST}"
    log "queued=${queued} skipped=${skipped}; waiting for jobs"
    gpu_queue_wait
}

aggregate_table() {
    "${PYTHON_BIN}" -m fsac.torgo.ger.aggregate_paper_table \
        --manifest "${MANIFEST}" \
        --out-csv "${TABLE_CSV}" \
        --out-md "${TABLE_MD}"
}

diff_outputs() {
    "${PYTHON_BIN}" - "${MANIFEST}" "${DIFF_REPORT}" "${DIFF_SUMMARY}" <<'PYEOF'
import csv
import json
import math
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
report = Path(sys.argv[2])
summary = Path(sys.argv[3])
SEVERITIES = ("Severe", "Moderately-severe", "Moderate", "Mild", "control")


def close(a, b, atol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)
    except Exception:
        return a == b


def load_pred(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            filename = (row.get("filename") or row.get("file") or "").strip()
            prediction = row.get("prediction") or ""
            rows.append((filename, prediction))
        return rows


def cmp_pred(gen_path, orig_path):
    if not gen_path.exists():
        return False, "GEN_MISSING"
    if not orig_path.exists():
        return False, "ORIG_MISSING"
    gen = load_pred(gen_path)
    orig = load_pred(orig_path)
    if gen == orig:
        return True, ""
    detail = f"len gen={len(gen)} orig={len(orig)}"
    for idx, (g, o) in enumerate(zip(gen, orig)):
        if g != o:
            detail += f" first-diff row{idx}: gen={g!r} orig={o!r}"
            break
    return False, detail


def cmp_json(gen_path, orig_path):
    if not gen_path.exists():
        return False, "GEN_MISSING"
    if not orig_path.exists():
        return False, "ORIG_MISSING"
    gen = json.loads(gen_path.read_text(encoding="utf-8"))
    orig = json.loads(orig_path.read_text(encoding="utf-8"))
    diffs = []
    for key in ("overall_wer", "overall_cer"):
        if not close(gen.get(key), orig.get(key)):
            diffs.append(f"{key}: gen={gen.get(key)} orig={orig.get(key)}")
    if gen.get("n_samples") != orig.get("n_samples"):
        diffs.append(f"n_samples: gen={gen.get('n_samples')} orig={orig.get('n_samples')}")
    gen_sev = gen.get("severity_breakdown") or {}
    orig_sev = orig.get("severity_breakdown") or {}
    for sev in SEVERITIES:
        gd = gen_sev.get(sev) or {}
        od = orig_sev.get(sev) or {}
        if not gd and not od:
            continue
        for metric in ("wer", "cer"):
            if not close(gd.get(metric), od.get(metric)):
                diffs.append(f"sev[{sev}].{metric}: gen={gd.get(metric)} orig={od.get(metric)}")
        if gd.get("n") != od.get("n"):
            diffs.append(f"sev[{sev}].n: gen={gd.get('n')} orig={od.get('n')}")
    return (not diffs), " | ".join(diffs[:12])


rows = []
with manifest.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        run_dir = Path(row["run_dir"])
        gen_dir = run_dir / f"whispering_llm{row['fsac_suffix']}"
        gen_pred = gen_dir / "predictions.csv"
        gen_json = gen_dir / "inference_results.json"
        orig_pred = Path(row["orig_pred"])
        orig_json = Path(row["orig_json"])
        pred_ok, pred_detail = cmp_pred(gen_pred, orig_pred)
        json_ok, json_detail = cmp_json(gen_json, orig_json)
        rows.append({
            "cell": row["cell"],
            "whisper": row["whisper"],
            "llm_tag": row["llm_tag"],
            "seed": row["seed"],
            "mode": row["mode"],
            "orig_mode": row["orig_mode"],
            "predictions_ok": pred_ok,
            "metrics_ok": json_ok,
            "pred_detail": pred_detail,
            "json_detail": json_detail,
            "generated_predictions": str(gen_pred),
            "original_predictions": str(orig_pred),
            "generated_results": str(gen_json),
            "original_results": str(orig_json),
        })

report.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "cell", "whisper", "llm_tag", "seed", "mode", "orig_mode",
    "predictions_ok", "metrics_ok", "pred_detail", "json_detail",
    "generated_predictions", "original_predictions", "generated_results",
    "original_results",
]
with report.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

n_total = len(rows)
n_pred = sum(1 for r in rows if r["predictions_ok"])
n_json = sum(1 for r in rows if r["metrics_ok"])
n_both = sum(1 for r in rows if r["predictions_ok"] and r["metrics_ok"])
bad = [r for r in rows if not (r["predictions_ok"] and r["metrics_ok"])]
lines = [
    f"personalized checkpoint-only reproduction: {n_total} cells",
    f"  predictions (filename+prediction equal): {n_pred:>3} / {n_total}",
    f"  inference_results numeric metrics equal: {n_json:>3} / {n_total}",
    f"  both checks passed                   : {n_both:>3} / {n_total}",
    f"  report                               : {report}",
]
if bad:
    lines.append("  first mismatches:")
    for row in bad[:10]:
        detail = row["pred_detail"] or row["json_detail"]
        lines.append(f"    {row['cell']}: {detail}")
else:
    lines.append(f"  --> all {n_total} cells reproduced")
summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if not bad else 2)
PYEOF
}

log "TORGO personalized checkpoint-only reproduction"
log "  checkpoints_root = ${CHECKPOINTS_ROOT}"
log "  fds_embeddings   = ${FDS_EMBEDDINGS_ROOT}"
log "  fsac_splits      = ${FSAC_SPLITS_DIR}"
log "  out_root         = ${OUT_ROOT}"
log "  grid             = whispers=[${WHISPERS}] llms=[${LLM_TAGS}] seeds=[${SEEDS}] modes=[${MODES}]"
log "  gpus             = ${GPUS}  python=${PYTHON_BIN}"

require_dir "${CHECKPOINTS_ROOT}"
require_dir "${FSAC_SPLITS_DIR}"
mkdir -p "${OUT_ROOT}"

gpu_preflight
stage_acoustic_cache
write_manifest
stage_run_inputs
if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1; manifest and staged inputs are ready, skipping inference/aggregate/diff"
    log "  manifest : ${MANIFEST}"
    exit 0
fi
run_jobs
aggregate_table
diff_outputs

log "done"
log "  manifest : ${MANIFEST}"
log "  diff     : ${DIFF_REPORT}"
log "  summary  : ${DIFF_SUMMARY}"
log "  table    : ${TABLE_CSV}  ${TABLE_MD}"
