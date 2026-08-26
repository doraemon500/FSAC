#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/paper_repro/emnlp2026_fdiv}"
PAPER_ROOT="${PAPER_ROOT:-$ROOT/paper/emnlp2026_FDiv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fsac_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

args=(
  -m fsac.torgo.ger.reproduce_emnlp2026_fdiv
  --out-root "$OUT_ROOT"
  --paper-root "$PAPER_ROOT"
)

if [[ -n "${PERSONALIZED_REPRO_ROOT:-}" ]]; then
  args+=(--personalized-root "$PERSONALIZED_REPRO_ROOT")
fi
if [[ -n "${LOSO_REPRO_ROOT:-}" ]]; then
  args+=(--loso-root "$LOSO_REPRO_ROOT")
fi
if [[ "${ALLOW_SOURCE_RESULTS:-0}" == "1" ]]; then
  args+=(--allow-source-results)
fi
if [[ -n "${TIMESTAMP:-}" ]]; then
  args+=(--timestamp "$TIMESTAMP")
fi

"$PYTHON_BIN" "${args[@]}"
