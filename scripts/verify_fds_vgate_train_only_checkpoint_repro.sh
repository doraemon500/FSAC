#!/bin/bash
# Checkpoint-only reproduction verifier for fds_vgate_train_only_main.
#
# This does not train GER checkpoints. It re-runs FSAC inference from the
# existing best_checkpoint.pt files and compares predictions/metrics against
# the original fds_vgate_train_only_main outputs.
#
# The original train-only VGate run used a per-seed FDS reference cache:
#   fds_cache_train_only/seed<S>/
# If that cache is not available, this script builds it under OUT_ROOT/_cache
# from the split JSON and Whisper-large-v2 frame embeddings. That is FDS
# preprocessing only, not GER model training.
#
# Example:
#   GPUS=0,1,2,3,4 bash scripts/verify_fds_vgate_train_only_checkpoint_repro.sh
#
# Useful overrides:
#   SOURCE_ROOT=results/torgo/runs/fds_vgate_train_only_main
#   WHISPER_CACHE_SOURCE_DIR=cache/whisper_embeddings/new
#   FDS_EMBEDDINGS_ROOT=cache/fds
#   TRAIN_ONLY_FDS_SOURCE_DIR=cache/whisper_embeddings/fds_cache_train_only
#   BUILD_FDS_CACHE=0
#   DRY_RUN=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source scripts/lib/gpu_queue.sh

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x /opt/anaconda3/bin/python ]]; then
    PYTHON_BIN="/opt/anaconda3/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: python/python3 not found in PATH" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/results/torgo/runs/fds_vgate_train_only_main}"
WHISPER_CACHE_SOURCE_DIR="${WHISPER_CACHE_SOURCE_DIR:-}"
FDS_EMBEDDINGS_ROOT="${FDS_EMBEDDINGS_ROOT:-${ROOT}/cache/fds}"
TRAIN_ONLY_FDS_SOURCE_DIR="${TRAIN_ONLY_FDS_SOURCE_DIR:-}"
FSAC_SPLITS_DIR="${FSAC_SPLITS_DIR:-${ROOT}/data/splits/torgo/personalized_v1_clean_no_bad_utt_v3}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/torgo/runs/repro_fds_vgate_train_only_checkpoint_${TS}}"
GPUS="${GPUS:-0}"
FDS_FIT_GPU="${FDS_FIT_GPU:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
BUILD_FDS_CACHE="${BUILD_FDS_CACHE:-1}"

export PYTHON_BIN
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPU_QUEUE_LOG_DIR="${OUT_ROOT}/_logs"
export GPU_QUEUE_FAIL_FAST="${GPU_QUEUE_FAIL_FAST:-1}"

MANIFEST="${OUT_ROOT}/_manifest.csv"
DIFF_REPORT="${OUT_ROOT}/_diff_report.csv"
DIFF_SUMMARY="${OUT_ROOT}/_diff_summary.txt"
FSAC_CACHE_DIR="${OUT_ROOT}/_cache"
ACOUSTIC_CACHE="${FSAC_CACHE_DIR}/whisper_embeddings/torgo_whisper_largev2"
TRAIN_ONLY_FDS_ROOT="${ACOUSTIC_CACHE}/fds_cache_train_only"

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
    local src=""
    if [[ -n "${WHISPER_CACHE_SOURCE_DIR}" ]]; then
        if [[ -d "${WHISPER_CACHE_SOURCE_DIR}/torgo_whisper_largev2" ]]; then
            src="${WHISPER_CACHE_SOURCE_DIR}/torgo_whisper_largev2"
        else
            src="${WHISPER_CACHE_SOURCE_DIR}"
        fi
    else
        src="${FDS_EMBEDDINGS_ROOT}/torgo_whisper_largev2"
    fi
    require_dir "${src}"

    mkdir -p "${ACOUSTIC_CACHE}"
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

    local train_only_src=""
    if [[ -n "${TRAIN_ONLY_FDS_SOURCE_DIR}" ]]; then
        train_only_src="${TRAIN_ONLY_FDS_SOURCE_DIR}"
    elif [[ -d "${src}/fds_cache_train_only" ]]; then
        train_only_src="${src}/fds_cache_train_only"
    elif [[ -d "${ROOT}/cache/whisper_embeddings/fds_cache_train_only" ]]; then
        train_only_src="${ROOT}/cache/whisper_embeddings/fds_cache_train_only"
    fi

    if [[ -n "${train_only_src}" ]]; then
        require_dir "${train_only_src}"
        mkdir -p "${TRAIN_ONLY_FDS_ROOT}"
        for seed_dir in "${train_only_src}"/seed*; do
            [[ -d "${seed_dir}" ]] || continue
            local dst="${TRAIN_ONLY_FDS_ROOT}/$(basename "${seed_dir}")"
            rm -rf "${dst}"
            ln -sfn "${seed_dir}" "${dst}"
        done
        log "linked existing train-only FDS cache from ${train_only_src}"
    fi
    log "staged acoustic cache: ${ACOUSTIC_CACHE} (${n_npz} speakers)"
}

write_manifest() {
    mkdir -p "${OUT_ROOT}"
    "${PYTHON_BIN}" - "${SOURCE_ROOT}" "${FSAC_SPLITS_DIR}" "${OUT_ROOT}" \
        "${TRAIN_ONLY_FDS_ROOT}" "${MANIFEST}" <<'PYEOF'
import csv
import re
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
splits_dir = Path(sys.argv[2])
out_root = Path(sys.argv[3])
train_only_fds_root = Path(sys.argv[4])
manifest = Path(sys.argv[5])

llm_models = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}

rows = []
for ckpt in sorted(source_root.glob("*/*/seed*/whispering_llm_vgate_s*/best_checkpoint.pt")):
    rel = ckpt.relative_to(source_root)
    whisper, llm_tag, seed_dir, run_name = rel.parts[:4]
    if llm_tag not in llm_models:
        raise SystemExit(f"unknown llm tag in checkpoint path: {ckpt}")
    m = re.fullmatch(r"seed(\d+)", seed_dir)
    if not m:
        raise SystemExit(f"bad seed dir in checkpoint path: {ckpt}")
    seed = m.group(1)
    expected_run = f"whispering_llm_vgate_s{seed}"
    if run_name != expected_run:
        raise SystemExit(f"bad run dir: expected {expected_run}, got {run_name}")

    seed_root = source_root / whisper / llm_tag / seed_dir
    orig_run_dir = seed_root / run_name
    prompts_csv = seed_root / "ger_prompts.csv"
    frame_dte = seed_root / "whisper_dte" / "frame_dte_scores.csv"
    split_file = splits_dir / f"seed{seed}.json"
    orig_json = orig_run_dir / "inference_results.json"
    orig_pred = orig_run_dir / "predictions.csv"
    fds_cache_dir = train_only_fds_root / f"seed{seed}"
    run_dir = out_root / whisper / llm_tag / seed_dir
    cell = f"{whisper}/{llm_tag}/{seed_dir}/vgate"

    missing = []
    for label, path in (
        ("ckpt", ckpt),
        ("prompts", prompts_csv),
        ("split", split_file),
        ("orig_json", orig_json),
        ("orig_pred", orig_pred),
    ):
        if not path.exists():
            missing.append(f"{label}={path}")
    if missing:
        raise SystemExit(f"missing inputs for {cell}:\n  " + "\n  ".join(missing))

    rows.append({
        "cell": cell,
        "whisper": whisper,
        "llm_tag": llm_tag,
        "llm_model": llm_models[llm_tag],
        "seed": seed,
        "mode": "vgate",
        "fsac_suffix": f"_vgate_s{seed}",
        "ckpt": str(ckpt),
        "prompts_csv": str(prompts_csv),
        "frame_dte": str(frame_dte) if frame_dte.exists() else "",
        "split_file": str(split_file),
        "split_tag": f"personalized_v1_clean_no_bad_utt_v3/seed{seed}",
        "fds_cache_dir": str(fds_cache_dir),
        "run_dir": str(run_dir),
        "orig_json": str(orig_json),
        "orig_pred": str(orig_pred),
    })

if not rows:
    raise SystemExit(f"no vgate checkpoints found under {source_root}")

manifest.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "cell", "whisper", "llm_tag", "llm_model", "seed", "mode",
    "fsac_suffix", "ckpt", "prompts_csv", "frame_dte", "split_file",
    "split_tag", "fds_cache_dir", "run_dir", "orig_json", "orig_pred",
]
with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote manifest ({len(rows)} cells): {manifest}")
PYEOF
}

build_missing_fds_caches() {
    "${PYTHON_BIN}" - "${MANIFEST}" "${BUILD_FDS_CACHE}" <<'PYEOF'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
build = sys.argv[2] == "1"
needed = {}
with manifest.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        needed[row["seed"]] = (Path(row["split_file"]), Path(row["fds_cache_dir"]))

missing = [
    (seed, split_file, cache_dir)
    for seed, (split_file, cache_dir) in sorted(needed.items(), key=lambda x: int(x[0]))
    if not (cache_dir / "raw_frame_scores.npz").exists()
]
if missing and not build:
    lines = [f"seed{seed}: {cache_dir}" for seed, _, cache_dir in missing]
    raise SystemExit("ERROR: missing train-only FDS caches and BUILD_FDS_CACHE=0:\n  " + "\n  ".join(lines))
for _, _, cache_dir in missing:
    cache_dir.mkdir(parents=True, exist_ok=True)
print("\n".join(f"{seed},{split_file},{cache_dir}" for seed, split_file, cache_dir in missing))
PYEOF
}

stage_run_inputs() {
    "${PYTHON_BIN}" - "${MANIFEST}" <<'PYEOF'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts_link = run_dir / "ger_prompts.csv"
    if prompts_link.exists() or prompts_link.is_symlink():
        prompts_link.unlink()
    prompts_link.symlink_to(Path(row["prompts_csv"]).resolve())

    if row.get("frame_dte"):
        src = Path(row["frame_dte"])
        if src.exists():
            dst_dir = run_dir / "whisper_dte"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "frame_dte_scores.csv"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

print(f"staged inputs for {len(rows)} cells")
PYEOF
}

build_infer_command() {
    local llm_model="$1" run_dir="$2" split_file="$3" split_tag="$4" seed="$5" \
          fsac_suffix="$6" ckpt="$7" fds_cache_dir="$8" label="$9"
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
export FSAC_FDS_CACHE_DIR="${fds_cache_dir}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
echo "[\$(date '+%H:%M:%S')] [job=${label}] START train-only-vgate checkpoint-only infer gpu=\${CUDA_VISIBLE_DEVICES:-?}"
"${PYTHON_BIN}" -m fsac.torgo.ger.infer_whispering_llm \\
  --model "${llm_model}" \\
  --checkpoint "${ckpt}" \\
  --prompts-csv "${run_dir}/ger_prompts.csv" \\
  --fds-mode vgate --real-frame-fds --fds-variant true \\
  --gpu 0 --seed "${seed}" \\
  --max-new-tokens 128 \\
  --risk-scale zscore --gate-arch mlp --gate-init 1.0 \\
  --attn-risk-mode v_only
COMMAND
}

run_fds_cache_jobs() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    local missing
    missing="$(build_missing_fds_caches)"
    if [[ -z "${missing}" ]]; then
        log "train-only FDS caches already available"
        return 0
    fi
    while IFS=, read -r seed split_file cache_dir; do
        [[ -n "${seed}" ]] || continue
        log "building train-only FDS cache seed=${seed} -> ${cache_dir}"
        CUDA_VISIBLE_DEVICES="${FDS_FIT_GPU}" \
        FSAC_CACHE_DIR="${FSAC_CACHE_DIR}" \
        WHISPER_MODEL="openai/whisper-large-v2" \
        "${PYTHON_BIN}" -m fsac.torgo.frame_fds_train_only \
            --split-file "${split_file}" \
            --out-dir "${cache_dir}" \
            --device cuda
    done <<< "${missing}"
}

run_jobs() {
    gpu_queue_init "${GPUS}"
    local queued=0 skipped=0
    while IFS=, read -r cell whisper llm_tag llm_model seed mode fsac_suffix ckpt \
        prompts_csv frame_dte split_file split_tag fds_cache_dir run_dir orig_json orig_pred; do
        [[ "${cell}" == "cell" ]] && continue
        orig_pred="${orig_pred%$'\r'}"
        require_file "${fds_cache_dir}/raw_frame_scores.npz"
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
              "${seed}" "${fsac_suffix}" "${ckpt}" "${fds_cache_dir}" "${cell}")"
        gpu_queue_submit "${cmd}" "${cell}"
        queued=$((queued + 1))
    done < "${MANIFEST}"
    log "queued=${queued} skipped=${skipped}; waiting for jobs"
    gpu_queue_wait
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
    "cell", "whisper", "llm_tag", "seed", "mode",
    "predictions_ok", "metrics_ok", "pred_detail", "json_detail",
    "generated_predictions", "original_predictions",
    "generated_results", "original_results",
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
    f"fds_vgate_train_only checkpoint-only reproduction: {n_total} cells",
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

log "TORGO fds_vgate_train_only checkpoint-only reproduction"
log "  source_root      = ${SOURCE_ROOT}"
log "  whisper_cache    = ${WHISPER_CACHE_SOURCE_DIR:-auto}"
log "  embeddings_root  = ${FDS_EMBEDDINGS_ROOT}"
log "  train_only_fds   = ${TRAIN_ONLY_FDS_SOURCE_DIR:-auto}"
log "  fsac_splits      = ${FSAC_SPLITS_DIR}"
log "  out_root         = ${OUT_ROOT}"
log "  gpus             = ${GPUS}  python=${PYTHON_BIN}"
log "  build_fds_cache  = ${BUILD_FDS_CACHE}"

require_dir "${SOURCE_ROOT}"
require_dir "${FSAC_SPLITS_DIR}"
mkdir -p "${OUT_ROOT}"

gpu_preflight
stage_acoustic_cache
write_manifest
stage_run_inputs
if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1; manifest and staged inputs are ready, skipping FDS build/inference/diff"
    log "  manifest : ${MANIFEST}"
    exit 0
fi
run_fds_cache_jobs
run_jobs
diff_outputs

log "done"
log "  manifest : ${MANIFEST}"
log "  diff     : ${DIFF_REPORT}"
log "  summary  : ${DIFF_SUMMARY}"
