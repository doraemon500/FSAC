from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jiwer import cer, wer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from fsac.models.whispering_llm import FSACWhisperingLLM
from fsac.torgo.config import (
    FDS_T,
    RESULT_DIR,
    SEVERITY_ORDER,
    WHISPER_EMBEDDING_DIR,
    WHISPER_ENCODER_DIM,
    WHISPER_FDS_RESULT_DIR,
    get_canonical_three_way_split,
)
from fsac.torgo.ger.train_ger import (
    _build_raw_frame_scores_cache,
    _compute_zscore_stats,
    _load_speaker_npz,
    _parse_risk_scale,
    _resolve_fds_cache_dir,
)


def _mode_suffix(args) -> str:
    suffix = os.environ.get("GER_ABLATION_SUFFIX")
    if suffix:
        return suffix
    seed = args.seed if args.seed is not None else 0
    if args.fds_mode == "plain":
        return f"_plain_s{seed}"
    if args.fds_mode == "fds_off":
        return f"_fds_off_s{seed}"
    if args.fds_mode == "vgate":
        return f"_vgate_s{seed}"
    raise ValueError(f"Unknown fds_mode: {args.fds_mode}")


def _adapter_mode_for_fds_mode(fds_mode: str) -> str:
    if fds_mode == "plain":
        return "plain"
    if fds_mode in {"fds_off", "vgate"}:
        return "v_gate"
    raise ValueError(f"Unknown fds_mode: {fds_mode}")


def normalise_text(text: str) -> str:
    text = str(text).upper().strip()
    text = re.sub(r"[^A-Z\s']", "", text)
    return " ".join(text.split())


class GERInferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        frame_fds_df: pd.DataFrame | None,
        raw_frame_scores: dict[str, np.ndarray] | None,
        risk_scale_mode: str,
        risk_scale_param: float | None,
        risk_zscore_stats: tuple[float, float] | None,
        use_frame_fds: bool,
        max_text_len: int,
        max_frames: int,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.raw_frame_scores = raw_frame_scores or {}
        self.risk_scale_mode = risk_scale_mode
        self.risk_scale_param = risk_scale_param
        self.risk_zscore_mean = risk_zscore_stats[0] if risk_zscore_stats else 0.0
        self.risk_zscore_std = risk_zscore_stats[1] if risk_zscore_stats else 1.0
        self.use_frame_fds = use_frame_fds
        self.max_text_len = max_text_len
        self.max_frames = max_frames
        self._speaker_cache: dict[str, dict] = {}

        self.utt_fds = {}
        if frame_fds_df is not None and not frame_fds_df.empty:
            mean_col = "frame_fds_mean" if "frame_fds_mean" in frame_fds_df.columns else "frame_risk_mean"
            self.utt_fds = {
                str(row["filename"]): float(row[mean_col])
                for _, row in frame_fds_df.iterrows()
            }

    def __len__(self) -> int:
        return len(self.df)

    def _scale_risk(self, raw_scores: torch.Tensor) -> torch.Tensor:
        if self.risk_scale_mode == "raw":
            return raw_scores / FDS_T
        if self.risk_scale_mode == "zscore":
            return (raw_scores - self.risk_zscore_mean) / max(self.risk_zscore_std, 1e-6)
        if self.risk_scale_mode == "ztemp":
            z = (raw_scores - self.risk_zscore_mean) / max(self.risk_zscore_std, 1e-6)
            return z * float(self.risk_scale_param)
        if self.risk_scale_mode == "temp":
            return (raw_scores / FDS_T) * float(self.risk_scale_param)
        return raw_scores / FDS_T

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        speaker = str(row["speaker"])
        prompt = str(row["prompt"])
        target_value = row.get("target", row.get("ref_text", ""))
        if target_value is None or (isinstance(target_value, float) and pd.isna(target_value)):
            target_value = ""
        target = str(target_value)
        sev_raw = row.get("severity", "")
        if sev_raw is None or (isinstance(sev_raw, float) and pd.isna(sev_raw)):
            severity = ""
        else:
            severity = str(sev_raw)

        whisper_frames = torch.zeros(self.max_frames, WHISPER_ENCODER_DIM, dtype=torch.float32)
        whisper_mask = torch.zeros(self.max_frames, dtype=torch.bool)
        frame_risks = torch.zeros(self.max_frames, dtype=torch.float32)

        speaker_map = _load_speaker_npz(self._speaker_cache, speaker)
        frames = speaker_map.get(filename)
        if frames is not None:
            n_valid = min(len(frames), self.max_frames)
            whisper_frames[:n_valid] = torch.from_numpy(frames[:n_valid]).float()
            whisper_mask[:n_valid] = True
            if self.use_frame_fds:
                if filename in self.raw_frame_scores:
                    raw = torch.from_numpy(self.raw_frame_scores[filename][:n_valid]).float()
                else:
                    raw = torch.full((n_valid,), float(self.utt_fds.get(filename, 0.0)))
                frame_risks[:n_valid] = self._scale_risk(raw)

        return {
            "filename": filename,
            "speaker": speaker,
            "severity": severity,
            "prompt": prompt,
            "target": target,
            "whisper_frames": whisper_frames,
            "frame_risks": frame_risks,
            "whisper_mask": whisper_mask,
        }


def _collate(batch: list[dict]) -> dict:
    tensor_keys = ["whisper_frames", "frame_risks", "whisper_mask"]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in ("filename", "speaker", "severity", "prompt", "target"):
        out[key] = [item[key] for item in batch]
    return out


def _load_base_model(model_name: str, device: torch.device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device)


def _make_model(args, ckpt: dict, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gating = ckpt.get("gating_config", {})
    gate_init_raw = gating.get("gate_init", args.gate_init)
    gate_init = None if gate_init_raw == "none" or gate_init_raw is None else float(gate_init_raw)
    base_lm = _load_base_model(args.model, device)
    model = FSACWhisperingLLM(
        base_lm=base_lm,
        d_whisper=args.d_whisper,
        adapter_bottleneck=args.bottleneck,
        adapter_n_heads=args.n_heads,
        wrap_every_n=args.wrap_every_n,
        gate_arch=gating.get("gate_arch", args.gate_arch),
        gate_init=gate_init,
        attn_risk_mode=gating.get("attn_risk_mode", args.attn_risk_mode),
        risk_scale=gating.get("risk_scale", args.risk_scale),
        adapter_mode=gating.get("adapter_mode", _adapter_mode_for_fds_mode(args.fds_mode)),
    ).to(device)
    model.load_checkpoint_state(ckpt)
    return tokenizer, model


def _severity_metrics(rows: list[dict]) -> dict:
    out = {}
    for sev in SEVERITY_ORDER:
        items = [r for r in rows if r["severity"] == sev]
        if not items:
            continue
        refs = [normalise_text(x["reference"]) for x in items]
        hyps = [normalise_text(x["prediction"]) for x in items]
        valid = [(r, h) for r, h in zip(refs, hyps) if r]
        if not valid:
            continue
        r_list, h_list = zip(*valid)
        out[sev] = {
            "wer": wer(list(r_list), list(h_list)),
            "cer": cer(list(r_list), list(h_list)),
            "n": len(valid),
        }

    ctrl_items = [r for r in rows if r["severity"] == "" or r["severity"] is None]
    if ctrl_items:
        refs = [normalise_text(x["reference"]) for x in ctrl_items]
        hyps = [normalise_text(x["prediction"]) for x in ctrl_items]
        valid = [(r, h) for r, h in zip(refs, hyps) if r]
        if valid:
            r_list, h_list = zip(*valid)
            out["control"] = {
                "wer": wer(list(r_list), list(h_list)),
                "cer": cer(list(r_list), list(h_list)),
                "n": len(valid),
            }
    return out


def infer(args) -> None:
    if args.fds_variant != "true":
        raise ValueError("Standalone FSAC main pipeline supports only --fds-variant true")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    out_dir = RESULT_DIR / f"whispering_llm{_mode_suffix(args)}"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    use_frame_fds = args.fds_mode == "vgate" and args.real_frame_fds

    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Mode: {args.fds_mode}; frame-level FDS enabled={use_frame_fds}")
    print(f"Acoustic/FDS Whisper cache: {WHISPER_EMBEDDING_DIR}")
    tokenizer, model = _make_model(args, ckpt, device)
    model.eval()

    prompts_csv = Path(args.prompts_csv) if args.prompts_csv else RESULT_DIR / "ger_prompts.csv"
    frame_fds_csv = WHISPER_FDS_RESULT_DIR / "frame_fds_scores.csv"
    prompts_df = pd.read_csv(prompts_csv)
    frame_fds_df = pd.read_csv(frame_fds_csv) if frame_fds_csv.exists() else None

    raw_frame_scores = {}
    risk_zscore_stats = ckpt.get("risk_zscore_stats")
    risk_scale_mode, risk_scale_param = _parse_risk_scale(ckpt.get("gating_config", {}).get("risk_scale", args.risk_scale))
    fds_cache_dir, fds_reference_tag = _resolve_fds_cache_dir()
    if use_frame_fds:
        print(f"FDS cache dir: {fds_cache_dir} (reference={fds_reference_tag})")
    if use_frame_fds:
        score_cache = fds_cache_dir / "raw_frame_scores.npz"
        raw_frame_scores = _build_raw_frame_scores_cache(score_cache, variant=args.fds_variant)
        if risk_scale_mode in {"zscore", "ztemp"} and not risk_zscore_stats:
            zscore_cache = fds_cache_dir / "zscore_stats.json"
            if zscore_cache.exists():
                payload = json.loads(zscore_cache.read_text(encoding="utf-8"))
                risk_zscore_stats = (float(payload["mean"]), float(payload["std"]))
                print(f"  Loaded z-score stats from {zscore_cache}: "
                      f"mean={risk_zscore_stats[0]:.3f}, std={risk_zscore_stats[1]:.3f}")
            else:
                risk_zscore_stats = _compute_zscore_stats(raw_frame_scores)

    _, _, test_files = get_canonical_three_way_split(prompts_df["filename"].unique())
    test_df = prompts_df[prompts_df["filename"].astype(str).isin(test_files)].reset_index(drop=True)
    dataset = GERInferenceDataset(
        test_df,
        tokenizer,
        frame_fds_df,
        raw_frame_scores,
        risk_scale_mode,
        risk_scale_param,
        tuple(risk_zscore_stats) if risk_zscore_stats else None,
        use_frame_fds,
        args.max_text_len,
        args.max_frames,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate)

    rows = []
    with torch.no_grad():
        for batch in loader:
            whisper_frames = batch["whisper_frames"].to(device)
            whisper_mask = batch["whisper_mask"].to(device)
            frame_risks = batch["frame_risks"].to(device) if use_frame_fds else None

            prompt_text = batch["prompt"][0]
            prompt_enc = tokenizer(
                prompt_text,
                truncation=True,
                max_length=args.max_text_len,
                return_tensors="pt",
            )
            prompt_ids = prompt_enc["input_ids"].to(device)

            generated = model.generate(
                input_ids=prompt_ids,
                whisper_frames=whisper_frames,
                frame_risks=frame_risks,
                whisper_mask=whisper_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
            )
            new_tokens = generated[0, prompt_ids.shape[1]:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            reference = batch["target"][0]

            ref_norm = normalise_text(reference)
            pred_norm = normalise_text(prediction)
            if not ref_norm:
                raise ValueError(f"Empty normalized reference: {batch['filename'][0]}")
            rows.append(
                {
                    "filename": batch["filename"][0],
                    "speaker": batch["speaker"][0],
                    "severity": batch["severity"][0],
                    "reference": reference,
                    "prediction": prediction,
                    "wer": wer(ref_norm, pred_norm),
                    "cer": cer(ref_norm, pred_norm),
                }
            )

    refs = [normalise_text(x["reference"]) for x in rows]
    hyps = [normalise_text(x["prediction"]) for x in rows]
    valid_pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if valid_pairs:
        r_list, h_list = zip(*valid_pairs)
        overall_wer = wer(list(r_list), list(h_list))
        overall_cer = cer(list(r_list), list(h_list))
    else:
        raise ValueError("No valid references available for evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    result = {
        "model_name": args.model,
        "fds_mode": args.fds_mode,
        "real_frame_fds": bool(use_frame_fds),
        "n_samples": len(rows),
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "severity_breakdown": _severity_metrics(rows),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "checkpoint": str(ckpt_path),
        "acoustic_fds_whisper_cache": str(WHISPER_EMBEDDING_DIR),
        "acoustic_fds_reference": fds_reference_tag,
        "acoustic_fds_cache_dir": str(fds_cache_dir),
    }
    (out_dir / "inference_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote predictions/results to {out_dir}")


def _normalize_args(args):
    if args.fds_mode is None:
        args.fds_mode = "vgate"
    if args.fds_variant is None:
        args.fds_variant = "true"
    return args


def main():
    parser = argparse.ArgumentParser(description="Run FSAC inference on TORGO.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompts-csv", type=str, default=None)
    parser.add_argument("--fds-mode", choices=["plain", "fds_off", "vgate"], default=None)
    parser.add_argument("--real-frame-fds", action="store_true")
    parser.add_argument("--fds-variant", choices=["true"], default=None)
    parser.add_argument("--risk-scale", type=str, default="raw")
    parser.add_argument("--gate-arch", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--gate-init", type=str, default="0.0")
    parser.add_argument("--attn-risk-mode", choices=["v_only"], default="v_only")
    parser.add_argument("--d-whisper", type=int, default=WHISPER_ENCODER_DIM)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-text-len", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--wrap-every-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = _normalize_args(parser.parse_args())
    infer(args)


if __name__ == "__main__":
    main()
