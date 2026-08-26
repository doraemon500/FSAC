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
export PYTHON_BIN

GPUS="${GPUS:-0,1,2,3}"
CACHE_FIT_GPU="${CACHE_FIT_GPU:-${GPUS%%,*}}"
SEEDS="${SEEDS:-42,123,456,789,999,2024,2025,3407,777,888}"
WHISPER_LLM_PAIRS="${WHISPER_LLM_PAIRS:-largev2:Qwen/Qwen3-8B,largev2:Qwen/Qwen3-4B,medium:Qwen/Qwen3-4B}"
FORCE="${FORCE:-0}"
SKIP_FDS_FIT="${SKIP_FDS_FIT:-0}"
PORT_BASE="${PORT_BASE:-33000}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-1}"
GA="${GA:-32}"
LR="${LR:-1e-4}"
LR_PROJ="${LR_PROJ:-1e-5}"
FDS_VARIANT="${FDS_VARIANT:-true}"

FSAC_CACHE_DIR="${FSAC_CACHE_DIR:-${ROOT}/cache}"
FSAC_RESULTS_DIR="${FSAC_RESULTS_DIR:-${ROOT}/results/torgo/runs/fds_vgate_train_only_main}"
FSAC_SPLITS_DIR="${FSAC_SPLITS_DIR:-${ROOT}/data/splits/torgo/personalized_v1_clean_no_bad_utt_v3}"
PAPER_TRAIN_DIR="${PAPER_TRAIN_DIR:-${ROOT}/data/paper_personalized_train}"
ACOUSTIC_WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL:-openai/whisper-large-v2}"

LARGEV2_CACHE_DIR="${FSAC_CACHE_DIR}/whisper_embeddings/torgo_whisper_largev2"
A_CACHE_DIR="${LARGEV2_CACHE_DIR}/fds_cache"
TRAIN_ONLY_FDS_ROOT="${LARGEV2_CACHE_DIR}/fds_cache_train_only"

export FSAC_CACHE_DIR FSAC_RESULTS_DIR
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

detect_torgo_root() {
    if [[ -n "${FSAC_DATA_DIR:-}" ]]; then
        echo "${FSAC_DATA_DIR}"
        return
    fi
    for c in "${ROOT}/data/torgo/raw" "${ROOT}/data/torgo/raw/TORGO"; do
        if [[ -d "${c}/F01" && -d "${c}/FC01" ]]; then
            echo "${c}"
            return
        fi
    done
}

FSAC_DATA_DIR="$(detect_torgo_root || true)"
if [[ -z "${FSAC_DATA_DIR}" ]]; then
    echo "ERROR: set FSAC_DATA_DIR to the TORGO directory containing F01/, FC01/, ..." >&2
    exit 1
fi
export FSAC_DATA_DIR

whisper_to_tag() {
    case "$1" in
        tiny|base|small|medium) echo "$1" ;;
        large-v2|largev2) echo "largev2" ;;
        *) echo "Unknown whisper key: $1" >&2; exit 1 ;;
    esac
}

llm_to_tag() {
    local m="$1"
    m="${m##*/}"
    echo "${m}" | tr '[:upper:].-' '[:lower:]__'
}

require_largev2_cache() {
    local missing=()
    for spk in FC01 FC02 FC03 MC01 MC02 F01 F03 F04 M01 M02 M03 M04 M05; do
        [[ -f "${LARGEV2_CACHE_DIR}/${spk}.npz" ]] || missing+=("${spk}")
    done
    if (( ${#missing[@]} == 0 )) && [[ "${FORCE}" != "1" ]]; then
        echo "[cache] Whisper-large-v2 speaker cache present"
        return
    fi
    echo "[cache] building Whisper-large-v2 speaker cache"
    CUDA_VISIBLE_DEVICES="${CACHE_FIT_GPU}" \
    FSAC_DATA_DIR="${FSAC_DATA_DIR}" \
    WHISPER_MODEL="${ACOUSTIC_WHISPER_MODEL}" \
        "${PYTHON_BIN}" -m fsac.torgo.extract_whisper_features \
            --model "${ACOUSTIC_WHISPER_MODEL}" \
            --device cuda
}

fit_all_control_fds() {
    if [[ -f "${A_CACHE_DIR}/raw_frame_scores.npz" && "${FORCE}" != "1" ]]; then
        echo "[fds] all-control cache present"
        return
    fi
    echo "[fds] fitting all-control FDS"
    CUDA_VISIBLE_DEVICES="${CACHE_FIT_GPU}" \
    FSAC_DATA_DIR="${FSAC_DATA_DIR}" \
    WHISPER_MODEL="${ACOUSTIC_WHISPER_MODEL}" \
        "${PYTHON_BIN}" -m fsac.torgo.frame_fds --device cuda
}

parse_grid() {
    IFS=',' read -r -a SEED_KEYS <<< "${SEEDS}"
    IFS=',' read -r -a PAIRS <<< "${WHISPER_LLM_PAIRS}"
    CELL_WHISPERS=()
    CELL_LLMS=()
    for pair in "${PAIRS[@]}"; do
        [[ -z "${pair}" ]] && continue
        [[ "${pair}" == *:* ]] || { echo "Bad WHISPER_LLM_PAIRS entry: ${pair}" >&2; exit 1; }
        CELL_WHISPERS+=("${pair%%:*}")
        CELL_LLMS+=("${pair#*:}")
    done
    N_PAIRS="${#CELL_WHISPERS[@]}"
    (( N_PAIRS > 0 )) || { echo "WHISPER_LLM_PAIRS is empty" >&2; exit 1; }
}

fit_train_only_fds() {
    mkdir -p "${TRAIN_ONLY_FDS_ROOT}"
    if [[ "${SKIP_FDS_FIT}" == "1" ]]; then
        echo "[fds] SKIP_FDS_FIT=1"
        return
    fi
    for seed in "${SEED_KEYS[@]}"; do
        local split_file="${FSAC_SPLITS_DIR}/seed${seed}.json"
        local out_dir="${TRAIN_ONLY_FDS_ROOT}/seed${seed}"
        [[ -f "${split_file}" ]] || { echo "Missing split: ${split_file}" >&2; exit 1; }
        if [[ -f "${out_dir}/raw_frame_scores.npz" && -f "${out_dir}/zscore_stats.json" && -f "${out_dir}/frame_fds_model.npz" && "${FORCE}" != "1" ]]; then
            echo "[fds] skip train-only seed${seed}"
            continue
        fi
        echo "[fds] train-only seed${seed}"
        CUDA_VISIBLE_DEVICES="${CACHE_FIT_GPU}" \
        WHISPER_MODEL="${ACOUSTIC_WHISPER_MODEL}" \
            "${PYTHON_BIN}" -m fsac.torgo.frame_fds_train_only \
                --split-file "${split_file}" \
                --out-dir "${out_dir}" \
                --device cuda
    done
}

prepare_run_dir() {
    local whisper_tag="$1" llm_tag="$2" seed="$3"
    local src="${PAPER_TRAIN_DIR}/${whisper_tag}/${llm_tag}/seed${seed}"
    local run_dir="${FSAC_RESULTS_DIR}/${whisper_tag}/${llm_tag}/seed${seed}"
    [[ -f "${src}/ger_prompts.csv" ]] || { echo "Missing prompts: ${src}/ger_prompts.csv" >&2; exit 1; }
    mkdir -p "${run_dir}/whisper_fds"
    cp -f "${src}/ger_prompts.csv" "${run_dir}/ger_prompts.csv"
    if [[ -f "${src}/whisper_dte/frame_dte_scores.csv" ]]; then
        mkdir -p "${run_dir}/whisper_dte"
        cp -f "${src}/whisper_dte/frame_dte_scores.csv" "${run_dir}/whisper_dte/frame_dte_scores.csv"
    fi
    echo "${run_dir}"
}

build_train_command() {
    local llm_model="$1" run_dir="$2" split_file="$3" seed="$4" suffix="$5" port="$6" label="$7" fds_cache_dir="$8"
    local prompts_csv="${run_dir}/ger_prompts.csv"
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
export WHISPER_MODEL="${ACOUSTIC_WHISPER_MODEL}"
export FSAC_FDS_CACHE_DIR="${fds_cache_dir}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
echo "[\$(date '+%H:%M:%S')] ${label} train"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=1 --master_port "${port}" \\
  -m fsac.torgo.ger.train_ger \\
  --model "${llm_model}" \\
  --prompts-csv "${prompts_csv}" \\
  --fds-mode vgate --real-frame-fds --fds-variant "${FDS_VARIANT}" \\
  --risk-scale zscore --gate-arch mlp --gate-init 1.0 \\
  --attn-risk-mode v_only \\
  --lr "${LR}" --lr-proj "${LR_PROJ}" \\
  --epochs "${EPOCHS}" --batch-size "${BS}" --grad-accum "${GA}" --no-wandb \\
  --seed "${seed}"
echo "[\$(date '+%H:%M:%S')] ${label} infer"
"${PYTHON_BIN}" -m fsac.torgo.ger.infer_whispering_llm \\
  --model "${llm_model}" \\
  --prompts-csv "${prompts_csv}" \\
  --fds-mode vgate --real-frame-fds --fds-variant "${FDS_VARIANT}" \\
  --gpu 0 --seed "${seed}" \\
  --max-new-tokens 128
EOF
}

run_train_grid() {
    export GPU_QUEUE_LOG_DIR="${FSAC_RESULTS_DIR}/_logs"
    gpu_queue_init "${GPUS}"
    local queued=0 skipped=0
    for seed in "${SEED_KEYS[@]}"; do
        local split_file="${FSAC_SPLITS_DIR}/seed${seed}.json"
        local fds_cache_dir="${TRAIN_ONLY_FDS_ROOT}/seed${seed}"
        [[ -f "${split_file}" ]] || { echo "Missing split: ${split_file}" >&2; exit 1; }
        [[ -f "${fds_cache_dir}/raw_frame_scores.npz" ]] || { echo "Missing FDS cache: ${fds_cache_dir}" >&2; exit 1; }
        for i in $(seq 0 $((N_PAIRS - 1))); do
            local whisper_tag llm_tag run_dir suffix out_json port label cmd
            whisper_tag="$(whisper_to_tag "${CELL_WHISPERS[$i]}")"
            llm_tag="$(llm_to_tag "${CELL_LLMS[$i]}")"
            run_dir="$(prepare_run_dir "${whisper_tag}" "${llm_tag}" "${seed}")"
            suffix="_vgate_s${seed}"
            out_json="${run_dir}/whispering_llm${suffix}/inference_results.json"
            if [[ -f "${out_json}" && "${FORCE}" != "1" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
            port=$((PORT_BASE + queued))
            label="${whisper_tag}/${llm_tag}/seed${seed}/vgate"
            cmd="$(build_train_command "${CELL_LLMS[$i]}" "${run_dir}" "${split_file}" "${seed}" "${suffix}" "${port}" "${label}" "${fds_cache_dir}")"
            gpu_queue_submit "${cmd}" "${label}"
            queued=$((queued + 1))
        done
    done
    echo "[train] queued=${queued} skipped=${skipped}"
    gpu_queue_wait
}

run_gate_fds() {
    export GPU_QUEUE_LOG_DIR="${FSAC_RESULTS_DIR}/_logs_gate_fds"
    gpu_queue_init "${GPUS}"
    local queued=0 skipped=0
    for seed in "${SEED_KEYS[@]}"; do
        local split_file="${FSAC_SPLITS_DIR}/seed${seed}.json"
        local fds_cache_dir="${TRAIN_ONLY_FDS_ROOT}/seed${seed}"
        for i in $(seq 0 $((N_PAIRS - 1))); do
            local whisper_tag llm_tag run_dir suffix ckpt out_corr label cmd
            whisper_tag="$(whisper_to_tag "${CELL_WHISPERS[$i]}")"
            llm_tag="$(llm_to_tag "${CELL_LLMS[$i]}")"
            run_dir="${FSAC_RESULTS_DIR}/${whisper_tag}/${llm_tag}/seed${seed}"
            suffix="_vgate_s${seed}"
            ckpt="${run_dir}/whispering_llm${suffix}/best_checkpoint.pt"
            out_corr="${run_dir}/whispering_llm${suffix}/gate_fds_corr.json"
            if [[ ! -f "${ckpt}" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
            if [[ -f "${out_corr}" && "${FORCE}" != "1" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
            label="gate_fds:${whisper_tag}/${llm_tag}/seed${seed}"
            cmd="cd \"${ROOT}\" && \"${PYTHON_BIN}\" -m fsac.torgo.ger.extract_gate_fds --run-dir \"${run_dir}\" --suffix \"${suffix}\" --split-file \"${split_file}\" --fds-cache-dir \"${fds_cache_dir}\" --device cuda"
            gpu_queue_submit "${cmd}" "${label}"
            queued=$((queued + 1))
        done
    done
    echo "[gate-fds] queued=${queued} skipped=${skipped}"
    gpu_queue_wait
}

echo "[cfg] results=${FSAC_RESULTS_DIR}"
echo "[cfg] paper_train=${PAPER_TRAIN_DIR}"
echo "[cfg] gpus=${GPUS}; seeds=${SEEDS}; pairs=${WHISPER_LLM_PAIRS}"

parse_grid
require_largev2_cache
fit_all_control_fds
fit_train_only_fds
run_train_grid
"${PYTHON_BIN}" -m fsac.torgo.ger.summarize_fds_vgate --result-root "${FSAC_RESULTS_DIR}" || true
run_gate_fds
"${PYTHON_BIN}" -m fsac.torgo.ger.compare_fds_reference_ablation \
    --setting-a-root "${PAPER_TRAIN_DIR}" \
    --plain-root "${PAPER_TRAIN_DIR}" \
    --train-only-results-root "${FSAC_RESULTS_DIR}" \
    --cache-root "${LARGEV2_CACHE_DIR}" \
    --seeds "${SEEDS}" \
    --pairs "${WHISPER_LLM_PAIRS}"

echo "[done] ${FSAC_RESULTS_DIR}"
