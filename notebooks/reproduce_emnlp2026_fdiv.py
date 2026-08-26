# %% [markdown]
# # EMNLP2026 FDiv Artifact Reproduction
#
# This notebook executes the reproduction pipeline from raw checkpoint-only
# inference artifacts. It does not start from completed paper tables.

# %%
from __future__ import annotations

import json
import importlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
for candidate in [
    ROOT,
    *ROOT.parents,
]:
    if (candidate / "fsac").is_dir():
        ROOT = candidate
        break
else:
    raise RuntimeError("Cannot find FSAC repository root. Start the kernel from the repo or install with `pip install -e .`.")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fsac.torgo.ger.reproduce_emnlp2026_fdiv as repro

repro = importlib.reload(repro)
PAPER_ROOT_DEFAULT = repro.PAPER_ROOT_DEFAULT
SOURCE_CONTROL_REF = repro.SOURCE_CONTROL_REF
SOURCE_LOSO = repro.SOURCE_LOSO
SOURCE_PERSONALIZED = repro.SOURCE_PERSONALIZED
aggregate_unified = repro.aggregate_unified
build_figures = repro.build_figures
build_report = repro.build_report
build_tables = repro.build_tables
build_unified_table = repro.build_unified_table
cells_from_manifest = repro.cells_from_manifest
cells_from_source = repro.cells_from_source
check_preconditions = repro.check_preconditions
latest_repro_root = repro.latest_repro_root
repo_display_path = repro.repo_display_path
validate_artifacts = repro.validate_artifacts
write_csv = repro.write_csv
write_text = repro.write_text


# %% [markdown]
# ## Configuration
#
# `ALLOW_SOURCE_RESULTS=False` is the strict mode: it requires completed FSAC
# checkpoint-only reproduction roots. Set it to `True` only to rebuild the paper
# artifacts directly from the original raw inference prediction/result files
# while the checkpoint-only repro grid is incomplete.

# %%
PAPER_ROOT = PAPER_ROOT_DEFAULT
PERSONALIZED_REPRO_ROOT = latest_repro_root("personalized")
LOSO_REPRO_ROOT = latest_repro_root("loso")
ALLOW_SOURCE_RESULTS = True
TIMESTAMP = datetime.utcnow().strftime("notebook_%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "results" / "paper_repro" / "emnlp2026_fdiv" / TIMESTAMP

print("ROOT:", ".")
print("PAPER_ROOT:", repo_display_path(PAPER_ROOT, ROOT))
print("SOURCE_PERSONALIZED:", repo_display_path(SOURCE_PERSONALIZED, ROOT))
print("SOURCE_LOSO:", repo_display_path(SOURCE_LOSO, ROOT))
print("SOURCE_CONTROL_REF:", repo_display_path(SOURCE_CONTROL_REF, ROOT))
print("PERSONALIZED_REPRO_ROOT:", repo_display_path(PERSONALIZED_REPRO_ROOT, ROOT))
print("LOSO_REPRO_ROOT:", repo_display_path(LOSO_REPRO_ROOT, ROOT))
print("ALLOW_SOURCE_RESULTS:", ALLOW_SOURCE_RESULTS)
print("OUT_DIR:", repo_display_path(OUT_DIR, ROOT))


# %% [markdown]
# ## Preflight
#
# The strict path fails early unless personalized is 300/300 and LOSO is
# 240/240. The source fallback still rebuilds artifacts from raw
# `predictions.csv`, `inference_results.json`, prompts, and FDS CSVs; it does
# not read completed paper tables.

# %%
if not ALLOW_SOURCE_RESULTS and (PERSONALIZED_REPRO_ROOT is None or LOSO_REPRO_ROOT is None):
    raise RuntimeError("Missing checkpoint-only repro roots. Run the verify scripts first.")

preconditions = check_preconditions(PERSONALIZED_REPRO_ROOT, LOSO_REPRO_ROOT, ALLOW_SOURCE_RESULTS)
display(pd.DataFrame(preconditions))

if any(row["status"] == "fail" for row in preconditions):
    raise RuntimeError("Strict checkpoint-only precondition failed. Re-run verification or set ALLOW_SOURCE_RESULTS=True for raw source-output fallback.")


# %% [markdown]
# ## Collect Raw Inference Cells
#
# This cell enumerates the raw per-cell files used for reproduction. RRQFormer
# and `.bak_*` artifacts are not collected.

# %%
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if ALLOW_SOURCE_RESULTS:
    cells = cells_from_source(SOURCE_PERSONALIZED, "personalized")
    cells += cells_from_source(SOURCE_LOSO, "loso")
else:
    cells = cells_from_manifest(PERSONALIZED_REPRO_ROOT, "personalized")
    cells += cells_from_manifest(LOSO_REPRO_ROOT, "loso")

cells = [c for c in cells if c.mode in {"plain", "fds_off", "vgate"} and c.pred.exists() and c.result_json.exists()]
manifest_rows = [dict(c.__dict__) for c in cells]
for row in manifest_rows:
    for key, value in list(row.items()):
        if isinstance(value, Path):
            row[key] = repo_display_path(value, ROOT)

write_csv(OUT_DIR / "_manifest.csv", manifest_rows)
write_text(
    OUT_DIR / "_manifest.json",
    json.dumps(
        {
            "personalized_root": repo_display_path(PERSONALIZED_REPRO_ROOT, ROOT),
            "loso_root": repo_display_path(LOSO_REPRO_ROOT, ROOT),
            "allow_source_results": ALLOW_SOURCE_RESULTS,
            "n_cells": len(cells),
        },
        indent=2,
    )
    + "\n",
)

manifest_df = pd.DataFrame(manifest_rows)
display(manifest_df.groupby(["dataset", "mode"]).size().rename("n_cells").reset_index())
display(manifest_df.head())


# %% [markdown]
# ## Build Unified Utterance-Seed Table
#
# This is the core recomputation step. It joins Plain/FDS-off/VGate predictions,
# recomputes WER components and Oracle@5 from prompts, attaches utterance FDS
# and DIV, then assigns headroom/FDS/DIV bins.

# %%
unified = build_unified_table(cells)
write_csv(OUT_DIR / "analysis" / "unified_utterance_seed_table.csv", unified)
write_csv(OUT_DIR / "analysis" / "personalized_setting_severity_summary.csv", aggregate_unified(unified, "personalized"))
write_csv(OUT_DIR / "analysis" / "loso_setting_severity_summary.csv", aggregate_unified(unified, "loso"))

unified_df = pd.DataFrame(unified)
print("unified rows:", len(unified_df))
display(unified_df[["dataset", "setting", "seed", "fold", "filename", "severity", "plain_wer", "vgate_wer", "oracle5_wer", "plain_headroom", "plain_vs_vgate_gain", "frame_fds_mean", "nbest_diversity"]].head())


# %% [markdown]
# ## Generate Paper Tables and Figures
#
# Tables are written as `.csv`, `.md`, and `.tex`. Figures are generated from
# the unified table or copied only when the paper uses a static asset.

# %%
tables = build_tables(OUT_DIR, unified, cells)
figures = build_figures(OUT_DIR, unified, PAPER_ROOT)
validation = validate_artifacts(
    OUT_DIR,
    PAPER_ROOT,
    preconditions,
    tables,
    figures,
    strict_paper_match=not ALLOW_SOURCE_RESULTS,
)
write_csv(OUT_DIR / "_validation_report.csv", validation)
build_report(OUT_DIR, validation, tables, figures)

display(pd.DataFrame(validation))


# %% [markdown]
# ## Display Reproduced Tables

# %%
for table_name in [
    "tab_main",
    "tab_headroom_bins",
    "tab_headroom_modsev",
    "tab_mechanism",
    "tab_fds_perturb",
    "tab_fds_scope_ablation",
    "tab_fds_scope_headroom",
    "tab_fds_frame_variation",
    "tab_loso_appendix",
    "tab_loso_headroom_bins",
    "tab_loso_headroom_speakers",
    "tab_params",
]:
    display(Markdown(f"### {table_name}"))
    display(pd.read_csv(OUT_DIR / "tables" / f"{table_name}.csv"))


# %% [markdown]
# ## Display Reproduced Figures

# %%
figure_checks = OUT_DIR / "analysis" / "figure_checks.csv"
if figure_checks.exists():
    display(Markdown("### figure_checks"))
    display(pd.read_csv(figure_checks))

for fig in [
    "fig_fds_vgate_clean.png",
    "fig_headroom_gain.pdf",
    "fig_fds_bin_gain.pdf",
    "fig_fds_gate_curve.pdf",
    "fig_prompt_example.pdf",
]:
    path = OUT_DIR / "figures" / fig
    display(Markdown(f"### {fig}"))
    print(repo_display_path(path, ROOT))
    preview = path.with_suffix(".png") if path.suffix.lower() == ".pdf" else path
    if path.suffix.lower() == ".pdf" and not preview.exists() and path.exists() and shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "180", str(path), str(path.with_suffix(""))], check=True)
    if preview.exists():
        display(Image(filename=str(preview)))
    else:
        display(Markdown(f"[Open figure]({repo_display_path(path, ROOT)})"))


# %% [markdown]
# ## Output Directory

# %%
print(repo_display_path(OUT_DIR, ROOT))
