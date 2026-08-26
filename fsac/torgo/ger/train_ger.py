from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from fsac.core.frame_fds import FrameLevelFDS
from fsac.models.whispering_llm import FSACWhisperingLLM
from fsac.torgo.config import (
    CONTROL_SPEAKERS,
    FDS_K,
    FDS_T,
    RESULT_DIR,
    WHISPER_EMBEDDING_DIR,
    WHISPER_ENCODER_DIM,
    WHISPER_FDS_RESULT_DIR,
    get_canonical_three_way_split,
)


def setup_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed training requested but torch.cuda.is_available() is False. "
                f"torch={getattr(torch, '__version__', 'unknown')} "
                f"torch.version.cuda={getattr(torch.version, 'cuda', None)} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}. "
                "Activate the CUDA-enabled environment/container before running torch.distributed."
            )
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_risk_scale(value: str) -> tuple[str, float | None]:
    value = value.lower()
    if value in {"raw", "zscore", "learnable"}:
        return value, None
    if value.startswith("temp_"):
        return "temp", float(value.split("_", 1)[1])
    if value.startswith("ztemp_"):
        return "ztemp", float(value.split("_", 1)[1])
    raise ValueError(f"Unknown risk scale: {value}")


def _load_speaker_npz(cache: dict[str, dict], speaker: str):
    if speaker not in cache:
        path = WHISPER_EMBEDDING_DIR / f"{speaker}.npz"
        if not path.exists():
            cache[speaker] = {}
        else:
            data = np.load(path, allow_pickle=True)
            cache[speaker] = {
                str(fname): data[f"frames_{idx}"]
                for idx, fname in enumerate(data["filenames"])
            }
    return cache[speaker]


def _resolve_fds_cache_dir() -> tuple[Path, str]:










    override = os.environ.get("FSAC_FDS_CACHE_DIR")
    if override:
        cache_dir = Path(override)
        manifest_path = cache_dir / "manifest.json"
        ref_tag = cache_dir.name
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                seed = payload.get("split_seed")
                kind = payload.get("kind", "fds_cache")
                if seed is not None:
                    ref_tag = f"{kind}_seed{seed}"
                else:
                    ref_tag = kind
            except (json.JSONDecodeError, OSError):
                pass
        return cache_dir, ref_tag
    return WHISPER_EMBEDDING_DIR / "fds_cache", "all_control"


def _load_raw_frame_scores_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    fnames = [str(x) for x in data["filenames"]]
    return {
        fname: data[f"scores_{idx}"].astype(np.float32)
        for idx, fname in enumerate(fnames)
    }


def _save_raw_frame_scores_cache(path: Path, mapping: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fnames = sorted(mapping)
    payload = {"filenames": np.asarray(fnames)}
    for idx, fname in enumerate(fnames):
        payload[f"scores_{idx}"] = mapping[fname].astype(np.float32)
    np.savez_compressed(str(path), **payload)


def _fit_or_load_fds_model(cache_dir: Path, device: str = "cpu") -> FrameLevelFDS:
    cache_path = cache_dir / "frame_fds_model.npz"
    model = FrameLevelFDS(K=FDS_K, T=FDS_T, device=device)
    if cache_path.exists():
        if is_main_process():
            print(f"Loading cached frame-level FDS model: {cache_path}")
        model.load(cache_path)
        return model

    if is_main_process():
        print("Fitting frame-level FDS model from control-speaker Whisper frames")
    all_ctrl = []
    for speaker in CONTROL_SPEAKERS:
        path = WHISPER_EMBEDDING_DIR / f"{speaker}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        for idx in range(len(data["filenames"])):
            all_ctrl.append(data[f"frames_{idx}"])
    if not all_ctrl:
        raise RuntimeError(f"No control Whisper frame cache found in {WHISPER_EMBEDDING_DIR}")

    ctrl_frames = np.concatenate(all_ctrl, axis=0)
    if ctrl_frames.shape[0] > 500_000:
        rng = np.random.RandomState(42)
        keep = rng.choice(ctrl_frames.shape[0], 500_000, replace=False)
        ctrl_frames = ctrl_frames[keep]
    model.fit(ctrl_frames)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.save(cache_path)
    if is_main_process():
        print(f"Saved frame-level FDS model: {cache_path}")
    return model


def _build_raw_frame_scores_cache(cache_path: Path, variant: str = "true") -> dict[str, np.ndarray]:
    if variant != "true":
        raise ValueError("Standalone FSAC main pipeline supports only --fds-variant true")
    if cache_path.exists():
        return _load_raw_frame_scores_cache(cache_path)

    cache_dir = cache_path.parent
    model = _fit_or_load_fds_model(cache_dir, device="cpu")
    mapping: dict[str, np.ndarray] = {}
    for spk_path in sorted(WHISPER_EMBEDDING_DIR.glob("*.npz")):
        data = np.load(str(spk_path), allow_pickle=True)
        for idx, fname in enumerate(data["filenames"]):
            frames = data[f"frames_{idx}"]
            mapping[str(fname)] = model.score_frames(frames).astype(np.float32)
    _save_raw_frame_scores_cache(cache_path, mapping)
    if is_main_process():
        print(f"Saved per-frame FDS cache: {cache_path} ({len(mapping)} utterances)")
    return mapping


def _compute_zscore_stats(raw_map: dict[str, np.ndarray]) -> tuple[float, float]:
    ctrl_scores = []
    for fname, scores in raw_map.items():
        speaker = Path(fname).name.split("_", 1)[0]
        if speaker in CONTROL_SPEAKERS:
            ctrl_scores.append(scores)
    if not ctrl_scores:
        return 0.0, 1.0
    values = np.concatenate(ctrl_scores).astype(np.float32)
    return float(values.mean()), float(values.std())


class GERMultimodalDataset(Dataset):
    def __init__(
        self,
        prompts_df: pd.DataFrame,
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
        self.df = prompts_df.reset_index(drop=True)
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

    def _load_frames(self, speaker: str, filename: str) -> np.ndarray | None:
        speaker_map = _load_speaker_npz(self._speaker_cache, speaker)
        return speaker_map.get(filename)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        speaker = str(row["speaker"])
        prompt = str(row["prompt"])
        target_value = row.get("target", row.get("ref_text", ""))
        if target_value is None or (isinstance(target_value, float) and pd.isna(target_value)):
            target_value = ""
        target = str(target_value)
        full_text = prompt + " " + target + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_text_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        whisper_frames = torch.zeros(self.max_frames, WHISPER_ENCODER_DIM, dtype=torch.float32)
        whisper_mask = torch.zeros(self.max_frames, dtype=torch.bool)
        frame_risks = torch.zeros(self.max_frames, dtype=torch.float32)

        frames = self._load_frames(speaker, filename)
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
            "prompt": prompt,
            "target": target,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "whisper_frames": whisper_frames,
            "frame_risks": frame_risks,
            "whisper_mask": whisper_mask,
        }


def _collate(batch: list[dict]) -> dict:
    keys = [
        "input_ids",
        "attention_mask",
        "labels",
        "whisper_frames",
        "frame_risks",
        "whisper_mask",
    ]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}
    for key in ("filename", "speaker", "prompt", "target"):
        out[key] = [item[key] for item in batch]
    return out


class SequenceCELoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.view(-1) != self.ignore_index
        if not valid_mask.any():
            return shift_logits.sum() * 0.0

        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return per_token_loss[valid_mask].mean()


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


def _load_base_model(model_name: str, device: torch.device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device)


def _make_model(args, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_lm = _load_base_model(args.model, device)
    gate_init = None if args.gate_init == "none" else float(args.gate_init)
    model = FSACWhisperingLLM(
        base_lm=base_lm,
        d_whisper=args.d_whisper,
        adapter_bottleneck=args.bottleneck,
        adapter_n_heads=args.n_heads,
        wrap_every_n=args.wrap_every_n,
        gate_arch=args.gate_arch,
        gate_init=gate_init,
        attn_risk_mode=args.attn_risk_mode,
        risk_scale=args.risk_scale,
        adapter_mode=_adapter_mode_for_fds_mode(args.fds_mode),
    ).to(device)
    return tokenizer, model


def train(args) -> None:
    import deepspeed

    set_seed(args.seed)
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_frame_fds = args.fds_mode == "vgate" and args.real_frame_fds
    risk_scale_mode, risk_scale_param = _parse_risk_scale(args.risk_scale)

    try:
        if is_main_process():
            print(f"Loading correction model: {args.model}")
            print(f"DeepSpeed ZeRO-2: rank={rank}, local_rank={local_rank}, world_size={world_size}")
            print(f"Mode: {args.fds_mode}; frame-level FDS enabled={use_frame_fds}")
            print(f"Acoustic/FDS Whisper cache: {WHISPER_EMBEDDING_DIR}")

        tokenizer, raw_model = _make_model(args, device)
        trainable, total = raw_model.count_trainable_params()
        if is_main_process():
            print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        optimizer = torch.optim.AdamW(
            [
                {"params": list(raw_model.whisper_proj.parameters()), "lr": args.lr_proj},
                {
                    "params": [
                        p
                        for wrapper in raw_model.wrapped_layers.values()
                        for p in list(wrapper.adapter_L.parameters()) + list(wrapper.adapter_W.parameters())
                    ],
                    "lr": args.lr,
                },
            ],
            weight_decay=0.01,
        )

        ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
            },
            "gradient_accumulation_steps": args.grad_accum,
            "gradient_clipping": 1.0,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "wall_clock_breakdown": False,
        }

        model, optimizer, _, _ = deepspeed.initialize(
            model=raw_model,
            optimizer=optimizer,
            config=ds_config,
        )

        prompts_csv = Path(args.prompts_csv) if args.prompts_csv else RESULT_DIR / "ger_prompts.csv"
        frame_fds_csv = WHISPER_FDS_RESULT_DIR / "frame_fds_scores.csv"
        prompts_df = pd.read_csv(prompts_csv)
        frame_fds_df = pd.read_csv(frame_fds_csv) if frame_fds_csv.exists() else None

        raw_frame_scores = {}
        risk_zscore_stats = None
        fds_cache_dir, fds_reference_tag = _resolve_fds_cache_dir()
        if use_frame_fds and is_main_process():
            print(f"FDS cache dir: {fds_cache_dir} (reference={fds_reference_tag})")
        if use_frame_fds:
            score_cache = fds_cache_dir / "raw_frame_scores.npz"
            raw_frame_scores = _build_raw_frame_scores_cache(score_cache, variant=args.fds_variant)
            if risk_scale_mode in {"zscore", "ztemp"}:
                zscore_cache = fds_cache_dir / "zscore_stats.json"
                if zscore_cache.exists():
                    payload = json.loads(zscore_cache.read_text(encoding="utf-8"))
                    risk_zscore_stats = (float(payload["mean"]), float(payload["std"]))
                else:
                    risk_zscore_stats = _compute_zscore_stats(raw_frame_scores)
                    zscore_cache.parent.mkdir(parents=True, exist_ok=True)
                    zscore_cache.write_text(
                        json.dumps({"mean": risk_zscore_stats[0], "std": risk_zscore_stats[1]}, indent=2),
                        encoding="utf-8",
                    )
                if is_main_process():
                    print(
                        f"FDS z-score stats: mean={risk_zscore_stats[0]:.3f}, "
                        f"std={risk_zscore_stats[1]:.3f}"
                    )

        train_files, dev_files, _ = get_canonical_three_way_split(prompts_df["filename"].unique())
        train_df = prompts_df[prompts_df["filename"].astype(str).isin(train_files)].reset_index(drop=True)
        dev_df = prompts_df[prompts_df["filename"].astype(str).isin(dev_files)].reset_index(drop=True)
        if is_main_process():
            print(f"Split rows: train={len(train_df)} dev={len(dev_df)}")

        train_ds = GERMultimodalDataset(
            train_df,
            tokenizer,
            frame_fds_df,
            raw_frame_scores,
            risk_scale_mode,
            risk_scale_param,
            risk_zscore_stats,
            use_frame_fds,
            args.max_text_len,
            args.max_frames,
        )
        dev_ds = GERMultimodalDataset(
            dev_df,
            tokenizer,
            frame_fds_df,
            raw_frame_scores,
            risk_scale_mode,
            risk_scale_param,
            risk_zscore_stats,
            use_frame_fds,
            args.max_text_len,
            args.max_frames,
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
        dev_sampler = DistributedSampler(dev_ds, shuffle=False) if world_size > 1 else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=_collate,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=dev_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=_collate,
        )

        loss_fn = SequenceCELoss()
        out_dir = RESULT_DIR / f"whispering_llm{_mode_suffix(args)}"
        if is_main_process():
            out_dir.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        history = []

        for epoch in range(args.epochs):
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            total_loss = 0.0
            n_steps = 0
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                whisper_frames = batch["whisper_frames"].to(device)
                whisper_mask = batch["whisper_mask"].to(device)
                frame_risks = batch["frame_risks"].to(device) if use_frame_fds else None

                outputs = model(
                    input_ids=input_ids,
                    whisper_frames=whisper_frames,
                    frame_risks=frame_risks,
                    whisper_mask=whisper_mask,
                    labels=labels,
                )
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                loss = loss_fn(logits, labels)

                if not torch.isfinite(loss):
                    finite_ratio = torch.isfinite(logits).float().mean().detach().item()
                    raise RuntimeError(
                        "Non-finite CE loss detected. "
                        f"fds_mode={args.fds_mode}, epoch={epoch + 1}, step={step}, "
                        f"logits_finite_ratio={finite_ratio:.6f}"
                    )

                model.backward(loss)
                model.step()

                total_loss += loss.item()
                n_steps += 1

            train_loss = total_loss / max(n_steps, 1)

            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    whisper_frames = batch["whisper_frames"].to(device)
                    whisper_mask = batch["whisper_mask"].to(device)
                    frame_risks = batch["frame_risks"].to(device) if use_frame_fds else None

                    outputs = model(
                        input_ids=input_ids,
                        whisper_frames=whisper_frames,
                        frame_risks=frame_risks,
                        whisper_mask=whisper_mask,
                        labels=labels,
                    )
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()
                    val_steps += 1

            if world_size > 1:
                val_loss_tensor = torch.tensor([val_loss, val_steps], device=device, dtype=torch.float32)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor[0].item() / max(val_loss_tensor[1].item(), 1)
            else:
                avg_val_loss = val_loss / max(val_steps, 1)

            if is_main_process():
                history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": avg_val_loss})
                print(
                    f"Epoch {epoch + 1}/{args.epochs}: "
                    f"train_loss={train_loss:.4f} val_loss={avg_val_loss:.4f}"
                )

                if avg_val_loss < best_val:
                    best_val = avg_val_loss
                    ckpt = raw_model.get_checkpoint_state()
                    ckpt.update(
                        {
                            "epoch": epoch + 1,
                            "val_loss": best_val,
                            "model_name": args.model,
                            "gating_config": {
                                "fds_mode": args.fds_mode,
                                "adapter_mode": _adapter_mode_for_fds_mode(args.fds_mode),
                                "gate_arch": args.gate_arch,
                                "gate_init": args.gate_init,
                                "attn_risk_mode": args.attn_risk_mode,
                                "risk_scale": args.risk_scale,
                                "real_frame_fds": bool(use_frame_fds),
                                "fds_variant": args.fds_variant,
                            },
                            "risk_zscore_stats": risk_zscore_stats,
                            "acoustic_fds_reference": fds_reference_tag,
                            "acoustic_fds_cache_dir": str(fds_cache_dir),
                        }
                    )
                    torch.save(ckpt, out_dir / "best_checkpoint.pt")
                    print(f"Saved checkpoint: {out_dir / 'best_checkpoint.pt'}")

        if is_main_process():
            (out_dir / "training_history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
    finally:
        cleanup_ddp()


def _normalize_args(args):
    if args.fds_mode is None:
        args.fds_mode = "vgate"
    if args.fds_variant is None:
        args.fds_variant = "true"
    return args


def main():
    parser = argparse.ArgumentParser(description="Train FSAC GER model on TORGO.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-proj", type=float, default=1e-5)
    parser.add_argument("--d-whisper", type=int, default=WHISPER_ENCODER_DIM)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-text-len", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--wrap-every-n", type=int, default=1)
    parser.add_argument("--prompts-csv", type=str, default=None)
    parser.add_argument("--fds-mode", choices=["plain", "fds_off", "vgate"], default=None)
    parser.add_argument("--real-frame-fds", action="store_true")
    parser.add_argument("--fds-variant", choices=["true"], default=None)
    parser.add_argument("--risk-scale", type=str, default="raw")
    parser.add_argument("--gate-arch", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--gate-init", type=str, default="0.0")
    parser.add_argument("--attn-risk-mode", choices=["v_only"], default="v_only")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="fsac-torgo")
    args = _normalize_args(parser.parse_args())
    train(args)


if __name__ == "__main__":
    main()
