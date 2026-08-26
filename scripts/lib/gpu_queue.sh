#!/bin/bash
# Per-GPU 1-job parallel queue helper.
#
# Usage:
#   source scripts/lib/gpu_queue.sh
#   gpu_queue_init 5        # 5 GPUs (slots 0..4)
#   gpu_queue_submit "<bash command>" "<label>"
#   gpu_queue_submit "<bash command>" "<label>"
#   ...
#   gpu_queue_wait          # block until all submitted jobs finish
#
# Each submitted command is launched with `CUDA_VISIBLE_DEVICES=<gpu>`
# bound to a free slot; submit() blocks while all slots are busy. Job
# stdout/stderr stream to the parent terminal (and to a per-job log file
# if `GPU_QUEUE_LOG_DIR` is exported before submit).
#
# `gpu_queue_init` may take a comma-separated list of physical GPU ids
# instead of a count, e.g. `gpu_queue_init "0,1,2,3,4"` or
# `gpu_queue_init 5` (auto-numbered 0..4).

set -uo pipefail

GPU_QUEUE_SIZE=0
GPU_QUEUE_FREE=()
declare -A GPU_QUEUE_PID2GPU=()
declare -A GPU_QUEUE_PID2LABEL=()

gpu_queue_init() {
    local arg="$1"
    GPU_QUEUE_FREE=()
    GPU_QUEUE_PID2GPU=()
    GPU_QUEUE_PID2LABEL=()
    if [[ "$arg" =~ , ]]; then
        IFS=',' read -r -a GPU_QUEUE_FREE <<< "$arg"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        local n="$arg"
        for ((i=0; i<n; i++)); do GPU_QUEUE_FREE+=("$i"); done
    else
        echo "[gpu_queue] init expects integer count or comma-list: got $arg" >&2
        return 1
    fi
    GPU_QUEUE_SIZE="${#GPU_QUEUE_FREE[@]}"
    echo "[gpu_queue] initialized with ${GPU_QUEUE_SIZE} GPU slot(s): ${GPU_QUEUE_FREE[*]}"
}

_gpu_queue_reap_finished() {
    local reaped=0
    for pid in "${!GPU_QUEUE_PID2GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local rc=0
            wait "$pid" 2>/dev/null || rc=$?
            local gpu="${GPU_QUEUE_PID2GPU[$pid]}"
            local label="${GPU_QUEUE_PID2LABEL[$pid]}"
            echo "[gpu_queue] reap pid=$pid gpu=$gpu rc=$rc label=$label"
            unset "GPU_QUEUE_PID2GPU[$pid]"
            unset "GPU_QUEUE_PID2LABEL[$pid]"
            GPU_QUEUE_FREE+=("$gpu")
            reaped=1
            if [[ $rc -ne 0 && "${GPU_QUEUE_FAIL_FAST:-0}" -eq 1 ]]; then
                echo "[gpu_queue] FAIL_FAST: aborting (label=$label rc=$rc)" >&2
                gpu_queue_kill_all
                return 2
            fi
        fi
    done
    return 0
}

gpu_queue_submit() {
    local cmd="$1"
    local label="${2:-job}"
    while [[ "${#GPU_QUEUE_FREE[@]}" -eq 0 ]]; do
        wait -n 2>/dev/null || true
        _gpu_queue_reap_finished || return $?
    done
    local gpu="${GPU_QUEUE_FREE[0]}"
    GPU_QUEUE_FREE=("${GPU_QUEUE_FREE[@]:1}")
    local logfile="/dev/null"
    if [[ -n "${GPU_QUEUE_LOG_DIR:-}" ]]; then
        mkdir -p "${GPU_QUEUE_LOG_DIR}"
        local safe_label="${label// /_}"
        safe_label="${safe_label//\//_}"
        logfile="${GPU_QUEUE_LOG_DIR}/$(date +%H%M%S)_${safe_label}.log"
    fi
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        if [[ "$logfile" != "/dev/null" ]]; then
            bash -c "$cmd" 2>&1 | tee "$logfile"
            exit "${PIPESTATUS[0]}"
        else
            bash -c "$cmd"
        fi
    ) &
    local pid=$!
    GPU_QUEUE_PID2GPU[$pid]="$gpu"
    GPU_QUEUE_PID2LABEL[$pid]="$label"
    echo "[gpu_queue] submit pid=$pid gpu=$gpu label=$label log=$logfile"
}

gpu_queue_wait() {
    while [[ "${#GPU_QUEUE_PID2GPU[@]}" -gt 0 ]]; do
        wait -n 2>/dev/null || true
        _gpu_queue_reap_finished || return $?
    done
    echo "[gpu_queue] all jobs finished"
}

gpu_queue_kill_all() {
    for pid in "${!GPU_QUEUE_PID2GPU[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${!GPU_QUEUE_PID2GPU[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
}

trap 'gpu_queue_kill_all' EXIT
