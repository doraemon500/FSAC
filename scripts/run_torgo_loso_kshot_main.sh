#!/bin/bash
# TORGO LOSO + k-shot calibration sweep (FSAC port of dte_research's
# run_torgo_loso_kshot.sh).
#
# Default grid:
#   Whisper {medium}                   x
#   LLM     {Qwen/Qwen3-4B}            x
#   Folds   {F01, F03, F04, M01-M05}   x
#   K       {0}                        x
#   Seeds   {42, 123, 456, 789, 999,
#            2024, 2025, 3407, 777, 888} x
#   Modes   {fds_off, vgate, plain}
#   = 240 cells.
#
# Output layout:
#   ${OUT_ROOT}/
#     _shared/<whisper_tag>/{ger_prompts.csv, nbest_hypotheses.csv,
#                             token_confidences.csv,
#                             whisper_fds/frame_fds_scores.csv}
#     <whisper_tag>/<llm_tag>/fold_<F>/k<K>/seed<S>/
#       ger_prompts.csv ...
#       whispering_llm_<mode>_s<S>/
#         best_checkpoint.pt + inference_results.json
#         training_history.json + predictions.csv
#     paper_table_loso_medium_qwen3_4b.{csv,md}
#     paper_table_loso_fold_breakdown.csv
#     _sanity_report.{txt,csv}
#     _logs/
#
# Reuses run_torgo_fds_vgate.sh's build_run_command via duplicated logic
# so we don't accidentally break the personalized_v1 wrapper. The split
# selection is delegated to FSAC's config._load_split_file which the
# TORGO_SPLIT_FILE env var hooks into.
#
# Usage:
#   GPUS=0,1,2,3,4 bash scripts/run_torgo_loso_kshot_main.sh
#
# Override env vars (defaults shown):
#   OUT_ROOT, SPLITS_DIR, SHARED_REUSE_ROOT, WHISPER_MODELS, LLM_MODELS,
#   FOLDS, KS, SEEDS, MODES, EPOCHS, BS, GA, LR, LR_PROJ, PORT_BASE,
#   FDS_VARIANT, FDS_ACOUSTIC_WHISPER_MODEL, FORCE
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "${ROOT}"
source scripts/lib/gpu_queue.sh

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x /opt/anaconda3/bin/python ]]; then
    PYTHON_BIN="/opt/anaconda3/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/torgo/runs/loso_kshot_medium_qwen3_4b_${TS}}"
SPLITS_DIR="${SPLITS_DIR:-${REPO_ROOT}/splits/torgo/loso_kshot_v2}"
SHARED_REUSE_ROOT="${SHARED_REUSE_ROOT:-${REPO_ROOT}/results/torgo/runs/personalized_v1/_shared}"
# FDS acoustic cache root: each <tag>/ subdir must contain per-speaker .npz
# files plus dte_cache/{raw_frame_scores.npz, zscore_stats.json}. The
# default matches the dte_research tree.
FDS_EMBEDDINGS_ROOT="${FDS_EMBEDDINGS_ROOT:-${REPO_ROOT}/embeddings}"
FDS_ACOUSTIC_WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL:-openai/whisper-large-v2}"
# Whisper tag used by FSAC config.py for the FDS acoustic cache directory.
# config.py builds WHISPER_EMBEDDING_DIR = ${FSAC_CACHE_DIR}/whisper_embeddings/torgo_whisper_<tag>
# from WHISPER_MODEL; FDS_ACOUSTIC_WHISPER_MODEL is the model we export to
# every train/infer subshell, so the tag here must match _whisper_tag()'s
# output for that model.
FDS_ACOUSTIC_TAG="${FDS_ACOUSTIC_TAG:-largev2}"

WHISPER_MODELS="${WHISPER_MODELS:-medium}"
LLM_MODELS="${LLM_MODELS:-Qwen/Qwen3-4B}"
FOLDS="${FOLDS:-F01,F03,F04,M01,M02,M03,M04,M05}"
KS="${KS:-0}"
SEEDS="${SEEDS:-42,123,456,789,999,2024,2025,3407,777,888}"
MODES="${MODES:-fds_off,vgate,plain}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-1}"
GA="${GA:-32}"
LR="${LR:-1e-4}"
LR_PROJ="${LR_PROJ:-1e-5}"
FDS_VARIANT="${FDS_VARIANT:-true}"
GPUS="${GPUS:-0}"
# Pinned PORT_BASE outside the 29xxx / 31xxx ranges used by older runners
# to avoid the EADDRINUSE races that hit the original loso_kshot_v3 sweep.
PORT_BASE="${PORT_BASE:-36000}"
FORCE="${FORCE:-0}"
FSAC_DATA_DIR="${FSAC_DATA_DIR:-${ROOT}/data/torgo/raw}"
FSAC_CACHE_DIR="${FSAC_CACHE_DIR:-${ROOT}/cache}"

export FSAC_DATA_DIR FSAC_CACHE_DIR
export GPU_QUEUE_LOG_DIR="${OUT_ROOT}/_logs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "LOSO + k-shot suite (FSAC port)"
log "  OUT_ROOT             = ${OUT_ROOT}"
log "  SPLITS_DIR           = ${SPLITS_DIR}"
log "  SHARED_REUSE_ROOT    = ${SHARED_REUSE_ROOT}"
log "  FDS_EMBEDDINGS_ROOT  = ${FDS_EMBEDDINGS_ROOT}"
log "  FDS acoustic         = ${FDS_ACOUSTIC_WHISPER_MODEL} (tag=${FDS_ACOUSTIC_TAG})"
log "  grid: whispers=[${WHISPER_MODELS}]  llms=[${LLM_MODELS}]"
log "        folds=[${FOLDS}]  k=[${KS}]  seeds=[${SEEDS}]  modes=[${MODES}]"
log "  hp:   epochs=${EPOCHS} bs=${BS} ga=${GA} lr=${LR} lr_proj=${LR_PROJ}"
log "  gpus=[${GPUS}]  port_base=${PORT_BASE}  python=${PYTHON_BIN}"

###############################################################################
# Step 0: stage ${OUT_ROOT}/_shared/<whisper_tag>/ from SHARED_REUSE_ROOT.
#
# The dte_research personalized_v1/_shared/<tag>/ tree has the canonical
# preprocessing outputs (ger_prompts.csv + nbest_hypotheses.csv +
# token_confidences.csv + whisper_dte/frame_dte_scores.csv) per whisper
# tag, independent of (llm, seed). FSAC expects frame_fds_scores.csv with
# renamed columns for the FSAC training and inference schema.
###############################################################################
"${PYTHON_BIN}" - "${OUT_ROOT}" "${SHARED_REUSE_ROOT}" "${WHISPER_MODELS}" <<'PYEOF'
import csv
import shutil
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
src_root = Path(sys.argv[2])
whispers = [s.strip() for s in sys.argv[3].split(",") if s.strip()]


def whisper_tag(name):
    return "largev2" if name in ("large-v2", "largev2") else name


_RENAME = {
    "frame_dte_mean": "frame_fds_mean",
    "frame_dte_max":  "frame_fds_max",
    "frame_dte_std":  "frame_fds_std",
}


def _rename_dte_csv(src: Path, dst: Path) -> None:
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [_RENAME.get(n, n) for n in (reader.fieldnames or [])]
        records = list(reader)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rec in records:
            writer.writerow({_RENAME.get(k, k): v for k, v in rec.items()})


staged = []
for w in whispers:
    wt = whisper_tag(w)
    src = src_root / wt
    if not src.is_dir():
        sys.exit(f"ERROR: missing shared dir for whisper={w}: {src}")
    shared = out_root / "_shared" / wt
    (shared / "whisper_fds").mkdir(parents=True, exist_ok=True)

    for name in ("ger_prompts.csv", "nbest_hypotheses.csv", "token_confidences.csv"):
        s = src / name
        if not s.is_file():
            sys.exit(f"ERROR: missing {s}")
        shutil.copy2(s, shared / name)

    dte_csv = src / "whisper_dte" / "frame_dte_scores.csv"
    if not dte_csv.is_file():
        sys.exit(f"ERROR: missing {dte_csv}")
    _rename_dte_csv(dte_csv, shared / "whisper_fds" / "frame_fds_scores.csv")
    staged.append(wt)

for wt in staged:
    print(f"  staged _shared/{wt}/  <- {src_root}/{wt}/")
print(f"[loso] _shared staging complete ({len(staged)} whispers)")
PYEOF

###############################################################################
# Step 0b: stage FSAC acoustic cache so vgate's frame-FDS fitter and the
# multimodal dataset can find per-speaker Whisper frames.
#
# Without this, train_ger.py's _fit_or_load_fds_model() crashes (vgate) and
# the fds_off / plain branches silently load zero acoustic embeddings, which
# previously caused the _DEPRECATED_plain_main_20260517_090411 run to
# produce text-only-GER artifacts mislabelled as "plain" results. NaN CE
# loss on the first batch is the LOSO symptom of the same misconfiguration.
#
# We symlink per-speaker .npz files and copy the small FDS cache. WHISPER_MODEL
# is pinned at job runtime so config.py resolves WHISPER_EMBEDDING_DIR to the
# staged directory.
###############################################################################
ACOUSTIC_SRC="${FDS_EMBEDDINGS_ROOT}/torgo_whisper_${FDS_ACOUSTIC_TAG}"
ACOUSTIC_DST="${FSAC_CACHE_DIR}/whisper_embeddings/torgo_whisper_${FDS_ACOUSTIC_TAG}"
log "Step 0b: staging acoustic cache"
log "  ${ACOUSTIC_SRC}  ->  ${ACOUSTIC_DST}"

if [[ ! -d "${ACOUSTIC_SRC}" ]]; then
    echo "ERROR: FDS acoustic source dir missing: ${ACOUSTIC_SRC}" >&2
    echo "  set FDS_EMBEDDINGS_ROOT to a directory containing torgo_whisper_${FDS_ACOUSTIC_TAG}/" >&2
    exit 1
fi
mkdir -p "${ACOUSTIC_DST}/fds_cache"
npz_count=0
for npz in "${ACOUSTIC_SRC}"/*.npz; do
    [[ -e "${npz}" ]] || continue
    ln -sfn "${npz}" "${ACOUSTIC_DST}/$(basename "${npz}")"
    npz_count=$((npz_count + 1))
done
if [[ "${npz_count}" -eq 0 ]]; then
    echo "ERROR: no .npz files found in ${ACOUSTIC_SRC}" >&2
    exit 1
fi
for fname in raw_frame_scores.npz zscore_stats.json; do
    src_fds="${ACOUSTIC_SRC}/dte_cache/${fname}"
    if [[ ! -f "${src_fds}" ]]; then
        echo "ERROR: missing FDS cache file: ${src_fds}" >&2
        exit 1
    fi
    cp -f "${src_fds}" "${ACOUSTIC_DST}/fds_cache/${fname}"
done
log "  staged ${npz_count} .npz symlinks + fds_cache/{raw_frame_scores.npz,zscore_stats.json}"

###############################################################################
# helpers (whisper / llm tagging mirrors run_torgo_fds_vgate.sh)
###############################################################################
whisper_to_tag() {
    case "$1" in
        tiny) echo "tiny" ;;
        base) echo "base" ;;
        small) echo "small" ;;
        medium) echo "medium" ;;
        large-v2|largev2) echo "largev2" ;;
        *) echo "Unknown whisper key: $1" >&2; exit 1 ;;
    esac
}

llm_to_tag() {
    local m="$1"
    m="${m##*/}"
    echo "${m}" | tr '[:upper:].-' '[:lower:]__'
}

prepare_run_dir() {
    local whisper_tag="$1" llm_tag="$2" fold="$3" k="$4" seed="$5"
    local shared="${OUT_ROOT}/_shared/${whisper_tag}"
    local run_dir="${OUT_ROOT}/${whisper_tag}/${llm_tag}/fold_${fold}/k${k}/seed${seed}"
    mkdir -p "${run_dir}/whisper_fds"
    cp -f "${shared}/ger_prompts.csv"                 "${run_dir}/ger_prompts.csv"
    cp -f "${shared}/nbest_hypotheses.csv"            "${run_dir}/nbest_hypotheses.csv"
    cp -f "${shared}/token_confidences.csv"           "${run_dir}/token_confidences.csv"
    cp -f "${shared}/whisper_fds/frame_fds_scores.csv" "${run_dir}/whisper_fds/frame_fds_scores.csv"
    echo "${run_dir}"
}

build_run_command() {
    local llm_model="$1" run_dir="$2" split_file="$3" split_tag="$4" \
          seed="$5" mode="$6" suffix="$7" port="$8" label="$9"
    local prompts_csv="${run_dir}/ger_prompts.csv"
    local train_mode_flag infer_mode_flag
    case "${mode}" in
        plain)
            train_mode_flag="--fds-mode plain"
            infer_mode_flag="--fds-mode plain"
            ;;
        fds_off)
            train_mode_flag="--fds-mode fds_off"
            infer_mode_flag="--fds-mode fds_off"
            ;;
        vgate)
            train_mode_flag="--fds-mode vgate --real-frame-fds"
            infer_mode_flag="--fds-mode vgate --real-frame-fds"
            ;;
        *) echo "Unknown mode: ${mode}" >&2; exit 1 ;;
    esac
    cat <<EOF
set -euo pipefail
cd "${ROOT}"
export FSAC_DATA_DIR="${FSAC_DATA_DIR}"
export FSAC_CACHE_DIR="${FSAC_CACHE_DIR}"
export FSAC_RESULTS_DIR="${OUT_ROOT}"
export TORGO_RESULT_DIR="${run_dir}"
export TORGO_SPLIT_FILE="${split_file}"
export TORGO_SPLIT_TAG="${split_tag}"
export GER_ABLATION_SUFFIX="${suffix}"
export WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
echo "[\$(date '+%H:%M:%S')] [job=${label}] START train  acoustic_fds_whisper=${FDS_ACOUSTIC_WHISPER_MODEL}"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=1 --master_port "${port}" \\
  -m fsac.torgo.ger.train_ger \\
  --model "${llm_model}" \\
  --prompts-csv "${prompts_csv}" \\
  ${train_mode_flag} --fds-variant "${FDS_VARIANT}" \\
  --risk-scale zscore --gate-arch mlp --gate-init 1.0 \\
  --attn-risk-mode v_only \\
  --lr "${LR}" --lr-proj "${LR_PROJ}" \\
  --epochs "${EPOCHS}" --batch-size "${BS}" --grad-accum "${GA}" --no-wandb \\
  --seed "${seed}"
echo "[\$(date '+%H:%M:%S')] [job=${label}] START infer"
"${PYTHON_BIN}" -m fsac.torgo.ger.infer_whispering_llm \\
  --model "${llm_model}" \\
  --prompts-csv "${prompts_csv}" \\
  ${infer_mode_flag} --fds-variant "${FDS_VARIANT}" --gpu 0 --seed "${seed}" \\
  --max-new-tokens 128
infer_json="${run_dir}/whispering_llm${suffix}/inference_results.json"
if [[ -f "\${infer_json}" ]]; then
    "${PYTHON_BIN}" - "\${infer_json}" "${label}" <<'PYINNER'
import datetime, json, sys
p, label = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p))
    ow = d.get("overall_wer")
    sb = d.get("severity_breakdown") or {}
    g = lambda k: (sb.get(k) or {}).get("wer")
    fmt = lambda v: "-" if v is None else f"{v*100:.2f}%"
    parts = [f"overall={fmt(ow)}"]
    for k in ("Severe", "Moderately-severe", "Moderate", "Mild", "control"):
        if k in sb:
            parts.append(f"{k}={fmt(g(k))}")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [RESULT] {label}  " + "  ".join(parts), flush=True)
except Exception as e:
    print(f"[RESULT] {label} parse_error: {e}", flush=True)
PYINNER
else
    echo "[RESULT] ${label} MISSING \${infer_json}"
fi
EOF
}

###############################################################################
# Step 1: enqueue jobs
###############################################################################
IFS=',' read -r -a WHISPER_KEYS <<< "${WHISPER_MODELS}"
IFS=',' read -r -a LLM_KEYS     <<< "${LLM_MODELS}"
IFS=',' read -r -a FOLD_KEYS    <<< "${FOLDS}"
IFS=',' read -r -a K_KEYS       <<< "${KS}"
IFS=',' read -r -a SEED_KEYS    <<< "${SEEDS}"
IFS=',' read -r -a MODE_KEYS    <<< "${MODES}"

gpu_queue_init "${GPUS}"
queued=0
skipped=0
missing_split=0

for whisper in "${WHISPER_KEYS[@]}"; do
    whisper_tag="$(whisper_to_tag "${whisper}")"
    for llm in "${LLM_KEYS[@]}"; do
        llm_tag="$(llm_to_tag "${llm}")"
        for fold in "${FOLD_KEYS[@]}"; do
            for k in "${K_KEYS[@]}"; do
                for seed in "${SEED_KEYS[@]}"; do
                    split_file="${SPLITS_DIR}/fold_${fold}/k${k}/seed${seed}.json"
                    if [[ ! -f "${split_file}" ]]; then
                        echo "[loso] WARN: missing split: ${split_file}" >&2
                        missing_split=$((missing_split + 1))
                        continue
                    fi
                    run_dir="$(prepare_run_dir "${whisper_tag}" "${llm_tag}" "${fold}" "${k}" "${seed}")"
                    split_tag="loso_kshot_v2/fold_${fold}/k${k}/seed${seed}"
                    for mode in "${MODE_KEYS[@]}"; do
                        suffix="_${mode}_s${seed}"
                        out_json="${run_dir}/whispering_llm${suffix}/inference_results.json"
                        label="${whisper}/${llm_tag}/fold_${fold}/k${k}/seed${seed}/${mode}"
                        if [[ -f "${out_json}" && "${FORCE}" != "1" ]]; then
                            echo "[loso] SKIP ${label}"
                            skipped=$((skipped + 1))
                            continue
                        fi
                        port=$((PORT_BASE + queued))
                        cmd="$(build_run_command "${llm}" "${run_dir}" "${split_file}" \
                                   "${split_tag}" "${seed}" "${mode}" "${suffix}" \
                                   "${port}" "${label}")"
                        gpu_queue_submit "${cmd}" "${label}"
                        queued=$((queued + 1))
                    done
                done
            done
        done
    done
done

echo "[loso] queued=${queued}  skipped=${skipped}  missing_split=${missing_split}"
gpu_queue_wait
echo "[loso] all jobs finished"

###############################################################################
# Step 2: aggregate paper table (LOSO layout)
###############################################################################
TABLE_BASE="${OUT_ROOT}/paper_table_loso_medium_qwen3_4b"
FOLD_TABLE="${OUT_ROOT}/paper_table_loso_fold_breakdown"
echo "[loso] aggregating LOSO paper table"
"${PYTHON_BIN}" -m fsac.torgo.ger.aggregate_loso_table \
    --scan-root "${OUT_ROOT}" \
    --out-csv  "${TABLE_BASE}.csv" \
    --out-md   "${TABLE_BASE}.md" \
    --fold-csv "${FOLD_TABLE}.csv" \
    --fold-md  "${FOLD_TABLE}.md" || echo "[loso] WARN: aggregator failed (non-fatal for sanity check)"

###############################################################################
# Step 3: sanity check
###############################################################################
SANITY_TXT="${OUT_ROOT}/_sanity_report.txt"
SANITY_CSV="${OUT_ROOT}/_sanity_report.csv"
"${PYTHON_BIN}" - "${OUT_ROOT}" "${WHISPER_MODELS}" "${LLM_MODELS}" "${FOLDS}" \
    "${KS}" "${SEEDS}" "${MODES}" "${SANITY_TXT}" "${SANITY_CSV}" <<'PYEOF'
import csv
import json
import math
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
whispers = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
llms = [s.strip() for s in sys.argv[3].split(",") if s.strip()]
folds = [s.strip() for s in sys.argv[4].split(",") if s.strip()]
ks = [s.strip() for s in sys.argv[5].split(",") if s.strip()]
seeds = [s.strip() for s in sys.argv[6].split(",") if s.strip()]
modes = [s.strip() for s in sys.argv[7].split(",") if s.strip()]
report_txt = Path(sys.argv[8])
report_csv = Path(sys.argv[9])


def whisper_tag(name):
    return "largev2" if name in ("large-v2", "largev2") else name


def llm_tag(model_id):
    short = model_id.split("/")[-1]
    return short.lower().replace(".", "_").replace("-", "_")


SEV_KEYS = ("Severe", "Moderately-severe", "Moderate", "Mild", "control")

rows = []
n_ok = n_missing = n_bad = 0
for w in whispers:
    wt = whisper_tag(w)
    for lm in llms:
        lt = llm_tag(lm)
        for fold in folds:
            for k in ks:
                for s in seeds:
                    for mode in modes:
                        run = out_root / wt / lt / f"fold_{fold}" / f"k{k}" / f"seed{s}" / f"whispering_llm_{mode}_s{s}"
                        inf = run / "inference_results.json"
                        pred = run / "predictions.csv"
                        label = f"{wt}/{lt}/fold_{fold}/k{k}/seed{s}/{mode}"
                        if not inf.exists() or not pred.exists():
                            rows.append({"cell": label, "status": "MISSING",
                                         "overall_wer": "", "overall_cer": "", "n_samples": "",
                                         "detail": f"inf_exists={inf.exists()} pred_exists={pred.exists()}"})
                            n_missing += 1
                            continue
                        try:
                            data = json.loads(inf.read_text(encoding="utf-8"))
                        except Exception as exc:
                            rows.append({"cell": label, "status": "BAD_JSON",
                                         "overall_wer": "", "overall_cer": "", "n_samples": "",
                                         "detail": str(exc)})
                            n_bad += 1
                            continue
                        ow = data.get("overall_wer")
                        oc = data.get("overall_cer")
                        ns = data.get("n_samples")
                        bad = []
                        for kk, v in (("overall_wer", ow), ("overall_cer", oc)):
                            if v is None:
                                bad.append(f"{kk}=None")
                            else:
                                try:
                                    if math.isnan(float(v)):
                                        bad.append(f"{kk}=NaN")
                                except Exception:
                                    bad.append(f"{kk}={v!r}")
                        if not isinstance(ns, int) or ns <= 0:
                            bad.append(f"n_samples={ns!r}")
                        sb = data.get("severity_breakdown") or {}
                        for sev in SEV_KEYS:
                            bin_d = sb.get(sev) or {}
                            for sub in ("wer", "cer"):
                                val = bin_d.get(sub)
                                if val is not None:
                                    try:
                                        if math.isnan(float(val)):
                                            bad.append(f"sev.{sev}.{sub}=NaN")
                                    except Exception:
                                        bad.append(f"sev.{sev}.{sub}={val!r}")
                        status = "OK" if not bad else "BAD_METRIC"
                        if status == "OK":
                            n_ok += 1
                        else:
                            n_bad += 1
                        rows.append({"cell": label, "status": status,
                                     "overall_wer": ow, "overall_cer": oc, "n_samples": ns,
                                     "detail": "; ".join(bad)})

with report_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["cell", "status", "overall_wer", "overall_cer", "n_samples", "detail"],
                            lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

n_total = len(rows)
lines = [
    f"LOSO main grid sanity report  ({n_total} cells)",
    f"  OK         : {n_ok:>3} / {n_total}",
    f"  MISSING    : {n_missing:>3} / {n_total}",
    f"  BAD        : {n_bad - max(0, n_bad - (n_total - n_ok - n_missing)):>3} / {n_total}",
    f"  csv        : {report_csv}",
]
bad_rows = [r for r in rows if r["status"] != "OK"]
if bad_rows:
    lines.append("  first 10 problem cells:")
    for r in bad_rows[:10]:
        lines.append(f"    {r['cell']}: {r['status']} {r['detail']}")
else:
    lines.append(f"  --> all {n_total} cells produced valid LOSO inference results")

report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
sys.exit(0 if not bad_rows else 2)
PYEOF
SANITY_EXIT=$?

echo
echo "[loso] complete"
echo "  paper table  : ${TABLE_BASE}.csv"
echo "  fold table   : ${FOLD_TABLE}.csv"
echo "  markdown     : ${TABLE_BASE}.md   ${FOLD_TABLE}.md"
echo "  sanity       : ${SANITY_TXT}"
echo "  cells root   : ${OUT_ROOT}"

if [[ "${SANITY_EXIT}" -ne 0 ]]; then
    echo "[loso] WARNING: sanity check reported issues (exit=${SANITY_EXIT})" >&2
fi
exit "${SANITY_EXIT}"
