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
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: python/python3 not found in PATH" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/torgo/runs/repro_all_paper_data_${TS}}"
GPUS="${GPUS:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_ARTIFACTS="${RUN_ARTIFACTS:-1}"
RUN_PERSONALIZED="${RUN_PERSONALIZED:-1}"
RUN_LOSO="${RUN_LOSO:-1}"
RUN_CONTROL_REF="${RUN_CONTROL_REF:-1}"
ALLOW_VERIFIER_MISMATCH="${ALLOW_VERIFIER_MISMATCH:-1}"

PERSONALIZED_CHECKPOINTS_ROOT="${PERSONALIZED_CHECKPOINTS_ROOT:-${ROOT}/results/torgo/runs/personalized_v1_promptablate_v3}"
LOSO_CHECKPOINTS_ROOT="${LOSO_CHECKPOINTS_ROOT:-${ROOT}/results/torgo/runs/loso_kshot_v3}"
CONTROL_SOURCE_ROOT="${CONTROL_SOURCE_ROOT:-${ROOT}/results/torgo/runs/fds_vgate_train_only_main}"
FDS_EMBEDDINGS_ROOT="${FDS_EMBEDDINGS_ROOT:-${ROOT}/cache/fds}"
CONTROL_WHISPER_CACHE_SOURCE_DIR="${CONTROL_WHISPER_CACHE_SOURCE_DIR:-}"
CONTROL_TRAIN_ONLY_FDS_SOURCE_DIR="${CONTROL_TRAIN_ONLY_FDS_SOURCE_DIR:-${ROOT}/cache/whisper_embeddings/fds_cache_train_only}"
CONTROL_BUILD_FDS_CACHE="${CONTROL_BUILD_FDS_CACHE:-1}"

PERSONALIZED_DATA_ROOT="${PERSONALIZED_DATA_ROOT:-${ROOT}/data/paper_personalized_train}"
LOSO_DATA_ROOT="${LOSO_DATA_ROOT:-${ROOT}/data/paper_loso_train}"
CONTROL_DATA_ROOT="${CONTROL_DATA_ROOT:-${ROOT}/data/paper_control_ref_ablation_train}"

EXACT_METRIC_ATOL="${EXACT_METRIC_ATOL:-1e-9}"
CONTROL_METRIC_ATOL="${CONTROL_METRIC_ATOL:-0.01}"
CONTROL_REQUIRE_METRIC_CLOSE="${CONTROL_REQUIRE_METRIC_CLOSE:-1}"

PERSONALIZED_OUT="${OUT_ROOT}/personalized"
LOSO_OUT="${OUT_ROOT}/loso"
CONTROL_OUT="${OUT_ROOT}/control_ref_ablation"
ARTIFACT_OUT="${OUT_ROOT}/paper_artifacts"
AUDIT_REPORT="${OUT_ROOT}/_paper_data_diff_report.csv"
AUDIT_SUMMARY="${OUT_ROOT}/_paper_data_summary.txt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
require_dir() { [[ -d "$1" ]] || { echo "ERROR: missing directory: $1" >&2; exit 1; }; }

run_verifier() {
    local label="$1"
    shift
    log "running ${label}"
    set +e
    "$@"
    local rc=$?
    set -e
    echo "${rc}" > "${OUT_ROOT}/_${label}_verifier_exit_code.txt"
    if [[ "${rc}" -ne 0 ]]; then
        log "WARN: ${label} verifier exited with rc=${rc}; data audit will still run"
        if [[ "${ALLOW_VERIFIER_MISMATCH}" != "1" ]]; then
            exit "${rc}"
        fi
    fi
}

write_run_config() {
    mkdir -p "${OUT_ROOT}"
    {
        echo "timestamp=${TS}"
        echo "out_root=${OUT_ROOT}"
        echo "gpus=${GPUS}"
        echo "run_inference=${RUN_INFERENCE}"
        echo "run_artifacts=${RUN_ARTIFACTS}"
        echo "run_personalized=${RUN_PERSONALIZED}"
        echo "run_loso=${RUN_LOSO}"
        echo "run_control_ref=${RUN_CONTROL_REF}"
        echo "personalized_data_root=${PERSONALIZED_DATA_ROOT}"
        echo "loso_data_root=${LOSO_DATA_ROOT}"
        echo "control_data_root=${CONTROL_DATA_ROOT}"
        echo "personalized_checkpoints_root=${PERSONALIZED_CHECKPOINTS_ROOT}"
        echo "loso_checkpoints_root=${LOSO_CHECKPOINTS_ROOT}"
        echo "control_source_root=${CONTROL_SOURCE_ROOT}"
        echo "exact_metric_atol=${EXACT_METRIC_ATOL}"
        echo "control_metric_atol=${CONTROL_METRIC_ATOL}"
    } > "${OUT_ROOT}/_run_config.txt"
}

run_artifact_repro() {
    [[ "${RUN_ARTIFACTS}" == "1" ]] || return 0
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY_RUN=1; skipping paper artifact CLI"
        return 0
    fi
    log "reproducing paper tables/figures from repo-local data"
    "${PYTHON_BIN}" -m fsac.torgo.ger.reproduce_emnlp2026_fdiv \
        --allow-source-results \
        --out-root "${ARTIFACT_OUT}" \
        --timestamp "paper_data_${TS}"
}

run_checkpoint_repro() {
    [[ "${RUN_INFERENCE}" == "1" ]] || return 0

    if [[ "${RUN_PERSONALIZED}" == "1" ]]; then
        run_verifier personalized \
            env PYTHON_BIN="${PYTHON_BIN}" GPUS="${GPUS}" FORCE="${FORCE}" DRY_RUN="${DRY_RUN}" \
            CHECKPOINTS_ROOT="${PERSONALIZED_CHECKPOINTS_ROOT}" \
            FDS_EMBEDDINGS_ROOT="${FDS_EMBEDDINGS_ROOT}" \
            OUT_ROOT="${PERSONALIZED_OUT}" \
            bash scripts/verify_personalized_checkpoint_repro.sh
    fi

    if [[ "${RUN_LOSO}" == "1" ]]; then
        run_verifier loso \
            env PYTHON_BIN="${PYTHON_BIN}" GPUS="${GPUS}" FORCE="${FORCE}" DRY_RUN="${DRY_RUN}" \
            CHECKPOINTS_ROOT="${LOSO_CHECKPOINTS_ROOT}" \
            FDS_EMBEDDINGS_ROOT="${FDS_EMBEDDINGS_ROOT}" \
            OUT_ROOT="${LOSO_OUT}" \
            bash scripts/verify_loso_checkpoint_repro.sh
    fi

    if [[ "${RUN_CONTROL_REF}" == "1" ]]; then
        run_verifier control_ref_ablation \
            env PYTHON_BIN="${PYTHON_BIN}" GPUS="${GPUS}" FORCE="${FORCE}" DRY_RUN="${DRY_RUN}" \
            SOURCE_ROOT="${CONTROL_SOURCE_ROOT}" \
            WHISPER_CACHE_SOURCE_DIR="${CONTROL_WHISPER_CACHE_SOURCE_DIR}" \
            FDS_EMBEDDINGS_ROOT="${FDS_EMBEDDINGS_ROOT}" \
            TRAIN_ONLY_FDS_SOURCE_DIR="${CONTROL_TRAIN_ONLY_FDS_SOURCE_DIR}" \
            BUILD_FDS_CACHE="${CONTROL_BUILD_FDS_CACHE}" \
            OUT_ROOT="${CONTROL_OUT}" \
            bash scripts/verify_fds_vgate_train_only_checkpoint_repro.sh
    fi
}

audit_against_paper_data() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY_RUN=1; skipping data audit"
        return 0
    fi
    "${PYTHON_BIN}" - \
        "${ROOT}" "${OUT_ROOT}" \
        "${PERSONALIZED_DATA_ROOT}" "${PERSONALIZED_OUT}" "${RUN_PERSONALIZED}" \
        "${LOSO_DATA_ROOT}" "${LOSO_OUT}" "${RUN_LOSO}" \
        "${CONTROL_DATA_ROOT}" "${CONTROL_OUT}" "${RUN_CONTROL_REF}" \
        "${EXACT_METRIC_ATOL}" "${CONTROL_METRIC_ATOL}" "${CONTROL_REQUIRE_METRIC_CLOSE}" \
        "${AUDIT_REPORT}" "${AUDIT_SUMMARY}" <<'PYEOF'
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
out_root = Path(sys.argv[2])
personalized_ref = Path(sys.argv[3])
personalized_gen = Path(sys.argv[4])
run_personalized = sys.argv[5] == "1"
loso_ref = Path(sys.argv[6])
loso_gen = Path(sys.argv[7])
run_loso = sys.argv[8] == "1"
control_ref = Path(sys.argv[9])
control_gen = Path(sys.argv[10])
run_control = sys.argv[11] == "1"
exact_atol = float(sys.argv[12])
control_atol = float(sys.argv[13])
control_require_metric_close = sys.argv[14] == "1"
report = Path(sys.argv[15])
summary = Path(sys.argv[16])

SEVERITIES = ("Severe", "Moderately-severe", "Moderate", "Mild", "control")
MODE_TO_FSAC = {"plain": "plain", "dte_off": "fds_off", "vgate": "vgate"}


def load_pred(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = []
        for row in csv.DictReader(f):
            filename = (row.get("filename") or row.get("file") or "").strip()
            prediction = row.get("prediction") or ""
            rows.append((filename, prediction))
        return rows


def cmp_pred(gen_path, ref_path):
    if not gen_path.exists():
        return False, "GEN_MISSING"
    if not ref_path.exists():
        return False, "REF_MISSING"
    gen = load_pred(gen_path)
    ref = load_pred(ref_path)
    if gen == ref:
        return True, ""
    detail = f"len gen={len(gen)} ref={len(ref)}"
    for idx, (g, r) in enumerate(zip(gen, ref)):
        if g != r:
            detail += f" first-diff row{idx}: gen={g!r} ref={r!r}"
            break
    return False, detail


def metric_pairs(data):
    yield "overall_wer", data.get("overall_wer")
    yield "overall_cer", data.get("overall_cer")
    sev = data.get("severity_breakdown") or {}
    for label in SEVERITIES:
        block = sev.get(label) or {}
        if not block:
            continue
        yield f"sev[{label}].wer", block.get("wer")
        yield f"sev[{label}].cer", block.get("cer")


def cmp_json(gen_path, ref_path, atol):
    if not gen_path.exists():
        return False, "GEN_MISSING", math.inf
    if not ref_path.exists():
        return False, "REF_MISSING", math.inf
    gen = json.loads(gen_path.read_text(encoding="utf-8"))
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    diffs = []
    max_abs = 0.0
    if gen.get("n_samples") != ref.get("n_samples"):
        diffs.append(f"n_samples: gen={gen.get('n_samples')} ref={ref.get('n_samples')}")
    gen_sev = gen.get("severity_breakdown") or {}
    ref_sev = ref.get("severity_breakdown") or {}
    for label in SEVERITIES:
        gd = gen_sev.get(label) or {}
        rd = ref_sev.get(label) or {}
        if not gd and not rd:
            continue
        if gd.get("n") != rd.get("n"):
            diffs.append(f"sev[{label}].n: gen={gd.get('n')} ref={rd.get('n')}")
    gen_metrics = dict(metric_pairs(gen))
    ref_metrics = dict(metric_pairs(ref))
    for key in sorted(set(gen_metrics) | set(ref_metrics)):
        gv = gen_metrics.get(key)
        rv = ref_metrics.get(key)
        if gv is None or rv is None:
            if gv != rv:
                diffs.append(f"{key}: gen={gv} ref={rv}")
            continue
        try:
            delta = abs(float(gv) - float(rv))
        except Exception:
            if gv != rv:
                diffs.append(f"{key}: gen={gv} ref={rv}")
            continue
        max_abs = max(max_abs, delta)
        if delta > atol:
            diffs.append(f"{key}: gen={gv} ref={rv} abs_diff={delta:.10g}")
    return (not diffs), " | ".join(diffs[:12]), max_abs


def add_row(rows, suite, strictness, cell, whisper, llm_tag, seed, mode,
            gen_json, ref_json, gen_pred, ref_pred, fold="", k=""):
    atol = exact_atol if strictness == "exact" else control_atol
    pred_ok, pred_detail = cmp_pred(gen_pred, ref_pred)
    metrics_ok, metric_detail, max_abs = cmp_json(gen_json, ref_json, atol)
    pred_present = gen_pred.exists() and ref_pred.exists()
    if strictness == "exact":
        accepted = pred_ok and metrics_ok
    elif control_require_metric_close:
        accepted = pred_present and metrics_ok
    else:
        accepted = pred_present
    rows.append({
        "suite": suite,
        "strictness": strictness,
        "cell": cell,
        "whisper": whisper,
        "llm_tag": llm_tag,
        "fold": fold,
        "k": k,
        "seed": seed,
        "mode": mode,
        "predictions_ok": pred_ok,
        "prediction_files_present": pred_present,
        "metrics_ok": metrics_ok,
        "accepted": accepted,
        "max_metric_abs_diff": "" if math.isinf(max_abs) else f"{max_abs:.12g}",
        "pred_detail": pred_detail,
        "metric_detail": metric_detail,
        "generated_predictions": str(gen_pred),
        "reference_predictions": str(ref_pred),
        "generated_results": str(gen_json),
        "reference_results": str(ref_json),
    })


def personalized_rows(rows):
    pattern = re.compile(r"whispering_llama_(plain|dte_off|vgate)_s(\d+)$")
    for ref_json in sorted(personalized_ref.glob("*/*/seed*/whispering_llama_*_s*/inference_results.json")):
        rel = ref_json.relative_to(personalized_ref)
        whisper, llm_tag, seed_dir, run_name = rel.parts[:4]
        m = pattern.fullmatch(run_name)
        if not m:
            continue
        orig_mode, seed = m.groups()
        fsac_mode = MODE_TO_FSAC[orig_mode]
        gen_dir = personalized_gen / whisper / llm_tag / seed_dir / fsac_mode / f"whispering_llm_{fsac_mode}_s{seed}"
        ref_pred = personalized_ref / whisper / llm_tag / seed_dir / f"{run_name}_predictions.csv"
        cell = f"{whisper}/{llm_tag}/{seed_dir}/{fsac_mode}"
        add_row(
            rows, "personalized", "exact", cell, whisper, llm_tag, seed, fsac_mode,
            gen_dir / "inference_results.json", ref_json,
            gen_dir / "predictions.csv", ref_pred,
        )


def loso_rows(rows):
    pattern = re.compile(r"whispering_llama_(plain|dte_off|vgate)_s(\d+)$")
    for ref_json in sorted(loso_ref.glob("*/*/fold_*/k*/seed*/whispering_llama_*_s*/inference_results.json")):
        rel = ref_json.relative_to(loso_ref)
        whisper, llm_tag, fold, k_dir, seed_dir, run_name = rel.parts[:6]
        m = pattern.fullmatch(run_name)
        if not m:
            continue
        orig_mode, seed = m.groups()
        fsac_mode = MODE_TO_FSAC[orig_mode]
        gen_dir = loso_gen / whisper / llm_tag / fold / k_dir / seed_dir / f"whispering_llm_{fsac_mode}_s{seed}"
        ref_pred = loso_ref / whisper / llm_tag / fold / k_dir / seed_dir / f"{run_name}_predictions.csv"
        cell = f"{whisper}/{llm_tag}/{fold}/{k_dir}/{seed_dir}/{fsac_mode}"
        add_row(
            rows, "loso", "exact", cell, whisper, llm_tag, seed, fsac_mode,
            gen_dir / "inference_results.json", ref_json,
            gen_dir / "predictions.csv", ref_pred,
            fold=fold, k=k_dir,
        )


def control_rows(rows):
    pattern = re.compile(r"whispering_llm_vgate_s(\d+)$")
    for ref_json in sorted(control_ref.glob("*/*/seed*/whispering_llm_vgate_s*/inference_results.json")):
        rel = ref_json.relative_to(control_ref)
        whisper, llm_tag, seed_dir, run_name = rel.parts[:4]
        m = pattern.fullmatch(run_name)
        if not m:
            continue
        seed = m.group(1)
        gen_dir = control_gen / whisper / llm_tag / seed_dir / run_name
        ref_pred = control_ref / whisper / llm_tag / seed_dir / f"{run_name}_predictions.csv"
        cell = f"{whisper}/{llm_tag}/{seed_dir}/vgate"
        add_row(
            rows, "control_ref_ablation", "approx", cell, whisper, llm_tag, seed, "vgate",
            gen_dir / "inference_results.json", ref_json,
            gen_dir / "predictions.csv", ref_pred,
        )


rows = []
if run_personalized:
    personalized_rows(rows)
if run_loso:
    loso_rows(rows)
if run_control:
    control_rows(rows)

report.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "suite", "strictness", "cell", "whisper", "llm_tag", "fold", "k", "seed", "mode",
    "predictions_ok", "prediction_files_present", "metrics_ok", "accepted", "max_metric_abs_diff",
    "pred_detail", "metric_detail",
    "generated_predictions", "reference_predictions",
    "generated_results", "reference_results",
]
with report.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

by_suite = defaultdict(list)
for row in rows:
    by_suite[row["suite"]].append(row)

lines = [
    "paper data checkpoint reproduction audit",
    f"  output root          : {out_root}",
    f"  report               : {report}",
    f"  exact metric atol    : {exact_atol}",
    f"  control metric atol  : {control_atol}",
]
failed = []
for suite in ("personalized", "loso", "control_ref_ablation"):
    suite_rows = by_suite.get(suite, [])
    if not suite_rows:
        continue
    total = len(suite_rows)
    pred = sum(1 for r in suite_rows if r["predictions_ok"])
    metric = sum(1 for r in suite_rows if r["metrics_ok"])
    accepted = sum(1 for r in suite_rows if r["accepted"])
    max_delta = max((float(r["max_metric_abs_diff"] or "0") for r in suite_rows), default=0.0)
    strictness = suite_rows[0]["strictness"]
    lines.append(
        f"  {suite:22s} {strictness:6s} cells={total:3d} "
        f"pred={pred:3d}/{total:3d} metric={metric:3d}/{total:3d} "
        f"accepted={accepted:3d}/{total:3d} max_metric_abs_diff={max_delta:.6g}"
    )
    bad = [r for r in suite_rows if not r["accepted"]]
    failed.extend(bad)
    if bad:
        lines.append(f"    first failures:")
        for row in bad[:8]:
            detail = row["metric_detail"] or row["pred_detail"]
            lines.append(f"      {row['cell']}: {detail}")

if run_control:
    lines.append("  control_ref_ablation uses approximate metric acceptance; prediction exact mismatch is reported but not required.")

if failed:
    lines.append(f"  --> FAILED accepted checks: {len(failed)}")
else:
    lines.append("  --> all required checks passed")

summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if not failed else 2)
PYEOF
}

log "paper data reproduction verification"
log "  out_root   = ${OUT_ROOT}"
log "  gpus       = ${GPUS}"
log "  data roots = ${PERSONALIZED_DATA_ROOT}, ${LOSO_DATA_ROOT}, ${CONTROL_DATA_ROOT}"
log "  control approximate metric atol = ${CONTROL_METRIC_ATOL}"

require_dir "${PERSONALIZED_DATA_ROOT}"
require_dir "${LOSO_DATA_ROOT}"
require_dir "${CONTROL_DATA_ROOT}"
write_run_config
run_artifact_repro
run_checkpoint_repro
audit_against_paper_data

log "done"
log "  report  : ${AUDIT_REPORT}"
log "  summary : ${AUDIT_SUMMARY}"
