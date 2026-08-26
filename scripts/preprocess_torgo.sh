#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

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
FDS_ACOUSTIC_WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL:-openai/whisper-large-v2}"
DEVICE="${DEVICE:-cuda}"
WHISPER_MODELS="${WHISPER_MODELS:-tiny,base,small,medium,large-v2}"
NBEST_N="${NBEST_N:-10}"
NBEST_BEAM="${NBEST_BEAM:-10}"

export FSAC_DATA_DIR FSAC_CACHE_DIR FSAC_RESULTS_DIR

whisper_to_name() {
    case "$1" in
        tiny) echo "openai/whisper-tiny.en" ;;
        base) echo "openai/whisper-base.en" ;;
        small) echo "openai/whisper-small.en" ;;
        medium) echo "openai/whisper-medium.en" ;;
        large-v2|largev2) echo "openai/whisper-large-v2" ;;
        *) echo "Unknown whisper key: $1" >&2; exit 1 ;;
    esac
}

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

strip_prompt_fds_block() {
    local in_csv="$1"
    local out_csv="$2"
    "${PYTHON_BIN}" - "${in_csv}" "${out_csv}" <<'PYEOF'
import csv
import json
import re
import sys
from pathlib import Path

in_csv = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
out_csv.parent.mkdir(parents=True, exist_ok=True)
pattern = re.compile(r"\nPronunciation risk:\n(?:- [^\n]+\n)+\n", re.DOTALL)

with in_csv.open(newline="", encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    rows = list(reader)
    fieldnames = reader.fieldnames or []

modified = 0
for row in rows:
    old = row["prompt"]
    new = pattern.sub("\n", old, count=1)
    modified += int(new != old)
    row["prompt"] = new

with out_csv.open("w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

report = {
    "input": str(in_csv),
    "output": str(out_csv),
    "rows": len(rows),
    "modified": modified,
    "remaining_risk_blocks": sum("Pronunciation risk:" in r["prompt"] for r in rows),
}
out_csv.with_suffix(out_csv.suffix + ".strip_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)
if report["remaining_risk_blocks"]:
    raise SystemExit(f"risk block stripping failed: {report}")
print(f"stripped prompt risk block: {modified}/{len(rows)} rows -> {out_csv}")
PYEOF
}

IFS=',' read -r -a WHISPER_KEYS <<< "${WHISPER_MODELS}"

echo "[preprocess] repo=${ROOT}"
echo "[preprocess] data=${FSAC_DATA_DIR}"
echo "[preprocess] cache=${FSAC_CACHE_DIR}"
echo "[preprocess] results=${FSAC_RESULTS_DIR}"
echo "[preprocess] acoustic FDS Whisper=${FDS_ACOUSTIC_WHISPER_MODEL}"

"${PYTHON_BIN}" - "${FSAC_DATA_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
speakers = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05", "FC01", "FC02", "FC03", "MC01", "MC02"]
total = sum(len(list((root / spk).glob("Session*/wav_headMic/*.wav"))) for spk in speakers)
if total == 0:
    raise SystemExit(
        f"No TORGO wav files found under FSAC_DATA_DIR={root}. "
        "Set FSAC_DATA_DIR to the raw TORGO root containing F01/Session*/wav_headMic/*.wav."
    )
print(f"[preprocess] raw_torgo_wavs={total}")
PY

for key in "${WHISPER_KEYS[@]}"; do
    model="$(whisper_to_name "${key}")"
    echo "[preprocess] extracting Whisper encoder cache: ${key} (${model})"
    WHISPER_MODEL="${model}" "${PYTHON_BIN}" -m fsac.torgo.extract_whisper_features \
        --model "${model}" --device "${DEVICE}"
done

needs_acoustic_extract=1
for key in "${WHISPER_KEYS[@]}"; do
    if [[ "$(whisper_to_name "${key}")" == "${FDS_ACOUSTIC_WHISPER_MODEL}" ]]; then
        needs_acoustic_extract=0
    fi
done
if [[ "${needs_acoustic_extract}" -eq 1 ]]; then
    echo "[preprocess] extracting acoustic FDS Whisper cache: ${FDS_ACOUSTIC_WHISPER_MODEL}"
    WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL}" "${PYTHON_BIN}" -m fsac.torgo.extract_whisper_features \
        --model "${FDS_ACOUSTIC_WHISPER_MODEL}" --device "${DEVICE}"
fi

ACOUSTIC_SHARED="${FSAC_RESULTS_DIR}/_shared/acoustic_largev2"
mkdir -p "${ACOUSTIC_SHARED}"
echo "[preprocess] building frame-level FDS cache from ${FDS_ACOUSTIC_WHISPER_MODEL}"
WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL}" TORGO_RESULT_DIR="${ACOUSTIC_SHARED}" \
    "${PYTHON_BIN}" -m fsac.torgo.frame_fds --device "${DEVICE}"

for key in "${WHISPER_KEYS[@]}"; do
    tag="$(whisper_to_tag "${key}")"
    model="$(whisper_to_name "${key}")"
    shared="${FSAC_RESULTS_DIR}/_shared/${tag}"
    mkdir -p "${shared}/whisper_fds"
    echo "[preprocess] generating text-side n-best/prompts: ${key} (${model})"
    WHISPER_MODEL="${model}" TORGO_RESULT_DIR="${shared}" \
        "${PYTHON_BIN}" -m fsac.torgo.generate_nbest \
        --model "${model}" --n "${NBEST_N}" --beam "${NBEST_BEAM}" --device "${DEVICE}"
    cp -f "${ACOUSTIC_SHARED}/whisper_fds/frame_fds_scores.csv" \
        "${shared}/whisper_fds/frame_fds_scores.csv"
    WHISPER_MODEL="${FDS_ACOUSTIC_WHISPER_MODEL}" TORGO_RESULT_DIR="${shared}" \
        "${PYTHON_BIN}" -m fsac.torgo.ger.fds_prompt
    mv "${shared}/ger_prompts.csv" "${shared}/ger_prompts_with_fds_prompt.csv"
    strip_prompt_fds_block "${shared}/ger_prompts_with_fds_prompt.csv" "${shared}/ger_prompts.csv"
done

echo "[preprocess] complete"
