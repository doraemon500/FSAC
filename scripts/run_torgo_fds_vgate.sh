#!/bin/bash
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
else
    PYTHON_BIN="python3"
fi
FSAC_DATA_DIR="${FSAC_DATA_DIR:-${ROOT}/data/torgo/raw}"
FSAC_CACHE_DIR="${FSAC_CACHE_DIR:-${ROOT}/cache}"
FSAC_RESULTS_DIR="${FSAC_RESULTS_DIR:-${ROOT}/results/torgo/runs/fds_vgate_main}"
FSAC_SPLITS_DIR="${FSAC_SPLITS_DIR:-${ROOT}/data/splits/torgo/personalized_v1_clean_no_bad_utt_v3}"
FDS_ACOUSTIC_WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL:-openai/whisper-large-v2}"
GPUS="${GPUS:-0}"
PORT_BASE="${PORT_BASE:-32000}"
WHISPER_MODELS="${WHISPER_MODELS:-tiny,base,small,medium,large-v2}"
LLM_MODELS="${LLM_MODELS:-Qwen/Qwen3-4B,Qwen/Qwen3-8B}"
SEEDS="${SEEDS:-42,123,456,789,999,2024,2025,3407,777,888}"
MODES="${MODES:-fds_off,vgate}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-1}"
GA="${GA:-32}"
LR="${LR:-1e-4}"
LR_PROJ="${LR_PROJ:-1e-5}"
FDS_VARIANT="${FDS_VARIANT:-true}"
FORCE="${FORCE:-0}"

export FSAC_DATA_DIR FSAC_CACHE_DIR FSAC_RESULTS_DIR
export GPU_QUEUE_LOG_DIR="${FSAC_RESULTS_DIR}/_logs"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    local whisper_tag="$1" llm_tag="$2" seed="$3"
    local shared="${FSAC_RESULTS_DIR}/_shared/${whisper_tag}"
    local run_dir="${FSAC_RESULTS_DIR}/${whisper_tag}/${llm_tag}/seed${seed}"
    [[ -f "${shared}/ger_prompts.csv" ]] || {
        echo "Missing ${shared}/ger_prompts.csv. Run scripts/preprocess_torgo.sh first." >&2
        exit 1
    }
    mkdir -p "${run_dir}/whisper_fds"
    cp -f "${shared}/ger_prompts.csv" "${run_dir}/ger_prompts.csv"
    cp -f "${shared}/nbest_hypotheses.csv" "${run_dir}/nbest_hypotheses.csv"
    cp -f "${shared}/token_confidences.csv" "${run_dir}/token_confidences.csv"
    cp -f "${shared}/whisper_fds/frame_fds_scores.csv" "${run_dir}/whisper_fds/frame_fds_scores.csv"
    echo "${run_dir}"
}

build_run_command() {
    local llm_model="$1" run_dir="$2" split_file="$3" seed="$4" mode="$5" suffix="$6" port="$7" label="$8"
    local prompts_csv="${run_dir}/ger_prompts.csv"
    local train_mode_flag
    local infer_mode_flag
    if [[ "${mode}" == "plain" ]]; then
        train_mode_flag="--fds-mode plain"
        infer_mode_flag="--fds-mode plain"
    elif [[ "${mode}" == "fds_off" ]]; then
        train_mode_flag="--fds-mode fds_off"
        infer_mode_flag="--fds-mode fds_off"
    elif [[ "${mode}" == "vgate" ]]; then
        train_mode_flag="--fds-mode vgate --real-frame-fds"
        infer_mode_flag="--fds-mode vgate --real-frame-fds"
    else
        echo "Unknown mode: ${mode}" >&2
        exit 1
    fi
    cat <<EOF
set -euo pipefail
cd "${ROOT}"
export FSAC_DATA_DIR="${FSAC_DATA_DIR}"
export FSAC_CACHE_DIR="${FSAC_CACHE_DIR}"
export FSAC_RESULTS_DIR="${FSAC_RESULTS_DIR}"
export TORGO_RESULT_DIR="${run_dir}"
export TORGO_SPLIT_FILE="${split_file}"
export TORGO_SPLIT_TAG="personalized_v1_clean_no_bad_utt_v3/seed${seed}"
export GER_ABLATION_SUFFIX="${suffix}"
export WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
echo "[\$(date '+%H:%M:%S')] [job=${label}] START train; acoustic_fds_whisper=${FDS_ACOUSTIC_WHISPER_MODEL}"
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
EOF
}

IFS=',' read -r -a WHISPER_KEYS <<< "${WHISPER_MODELS}"
IFS=',' read -r -a LLM_KEYS <<< "${LLM_MODELS}"
IFS=',' read -r -a SEED_KEYS <<< "${SEEDS}"
IFS=',' read -r -a MODE_KEYS <<< "${MODES}"

echo "[run] repo=${ROOT}"
echo "[run] results=${FSAC_RESULTS_DIR}"
echo "[run] splits=${FSAC_SPLITS_DIR}"
echo "[run] GPUs=${GPUS}"
echo "[run] acoustic FDS Whisper=${FDS_ACOUSTIC_WHISPER_MODEL}"

gpu_queue_init "${GPUS}"
queued=0
skipped=0

for whisper in "${WHISPER_KEYS[@]}"; do
    whisper_tag="$(whisper_to_tag "${whisper}")"
    for llm in "${LLM_KEYS[@]}"; do
        llm_tag="$(llm_to_tag "${llm}")"
        for seed in "${SEED_KEYS[@]}"; do
            split_file="${FSAC_SPLITS_DIR}/seed${seed}.json"
            [[ -f "${split_file}" ]] || { echo "Missing split ${split_file}" >&2; exit 1; }
            run_dir="$(prepare_run_dir "${whisper_tag}" "${llm_tag}" "${seed}")"
            for mode in "${MODE_KEYS[@]}"; do
                case "${mode}" in
                    plain) suffix="_plain_s${seed}" ;;
                    fds_off) suffix="_fds_off_s${seed}" ;;
                    vgate) suffix="_vgate_s${seed}" ;;
                    *) echo "Unknown mode: ${mode}" >&2; exit 1 ;;
                esac
                out_json="${run_dir}/whispering_llm${suffix}/inference_results.json"
                if [[ -f "${out_json}" && "${FORCE}" != "1" ]]; then
                    echo "[run] SKIP ${whisper}/${llm_tag}/seed${seed}/${mode}"
                    skipped=$((skipped + 1))
                    continue
                fi
                port=$((PORT_BASE + queued))
                label="${whisper}/${llm_tag}/seed${seed}/${mode}"
                cmd="$(build_run_command "${llm}" "${run_dir}" "${split_file}" "${seed}" "${mode}" "${suffix}" "${port}" "${label}")"
                gpu_queue_submit "${cmd}" "${label}"
                queued=$((queued + 1))
            done
        done
    done
done

echo "[run] queued=${queued} skipped=${skipped}"
gpu_queue_wait
"${PYTHON_BIN}" -m fsac.torgo.ger.summarize_fds_vgate --result-root "${FSAC_RESULTS_DIR}"
echo "[run] complete"
