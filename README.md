# Not All Acoustic Frames Contribute Equally: Frame-Selective Acoustic Conditioning for Dysarthric ASR Correction

Official implementation of our Findings of EMNLP 2026 paper on dysarthric ASR correction using the TORGO corpus.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

TORGO audio is expected under `data/torgo/raw/`. Override it with:

```bash
export FSAC_DATA_DIR=/path/to/TORGO
```

## Paper Artifacts

The paper tables and figures are reproduced from repo-local inference artifacts:

```text
data/paper_personalized_train/
data/paper_loso_train/
data/paper_control_ref_ablation_train/
```

These directories keep prompts, frame-DTE/FDS scores, inference metrics, training-loss histories, and prediction CSVs. Full checkpoints are not shipped in these `paper_*_train` directories, except for the two already retained personalized checkpoint files. The paper values can still be audited from the saved predictions and metrics.

Run the notebook:

```text
notebooks/reproduce_emnlp2026_fdiv.ipynb
```


## Checkpoint Inference Verification

When checkpoints are available, use the FSAC inference pipeline without retraining:

```bash
export GPUS=0
bash scripts/verify_all_paper_data_repro.sh
```

Use a comma-separated list for multi-GPU machines:

```bash
GPUS=0,1 bash scripts/verify_personalized_checkpoint_repro.sh
GPUS=0,1 bash scripts/verify_loso_checkpoint_repro.sh
GPUS=0,1 bash scripts/verify_fds_vgate_train_only_checkpoint_repro.sh
```

## Training

Preprocess TORGO and build Whisper/FDS caches:

```bash
WHISPER_MODELS=tiny,base,small,medium,large-v2 \
FDS_ACOUSTIC_WHISPER_MODEL=openai/whisper-large-v2 \
bash scripts/preprocess_torgo.sh
```

Train the personalized grid:

```bash
GPUS=0 bash scripts/run_torgo_fds_vgate.sh
```

Train the LOSO k-shot grid:

```bash
GPUS=0 bash scripts/run_torgo_loso_kshot_main.sh
```

Train the control-reference FDS ablation:

```bash
GPUS=0 bash scripts/run_fds_reference_split_ablation.sh
```

Set `GPUS=0,1,...` to parallelize on available local GPUs.

Single-cell entrypoints:

```bash
python -m fsac.torgo.ger.train_ger --model Qwen/Qwen3-4B --fds-mode plain
python -m fsac.torgo.ger.train_ger --model Qwen/Qwen3-4B --fds-mode fds_off
python -m fsac.torgo.ger.train_ger --model Qwen/Qwen3-4B --fds-mode vgate --real-frame-fds
python -m fsac.torgo.ger.infer_whispering_llm --model Qwen/Qwen3-4B --fds-mode vgate --real-frame-fds
```

## FDS Provenance

Frame deviation score (FDS) uses the non-parametric diffusion-time estimator from Livernoche et al., "On Diffusion Modeling for Anomaly Detection" (ICLR 2024). In this repo, the DTE-NP kNN inverse-gamma score is applied to Whisper frame features and used as a per-frame acoustic deviation signal.

The main VGate comparison keeps the acoustic injection path fixed and changes only whether frame-level FDS is used by the VGate adapter. `plain` is the no-FDS cross-attention adapter, `fds_off` keeps the VGate adapter without frame risk, and `vgate` enables frame-level FDS.
