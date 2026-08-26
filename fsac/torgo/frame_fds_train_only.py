from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from fsac.core.frame_fds import FrameLevelFDS
from fsac.torgo.config import (
    CONTROL_SPEAKERS,
    DYSARTHRIC_SPEAKERS,
    FDS_K,
    FDS_T,
    FDS_SEED,
    WHISPER_EMBEDDING_DIR,
)


def _load_split(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "train" not in payload:
        raise ValueError(f"Split file missing 'train' key: {path}")
    return payload


def _filename_to_speaker(name: str) -> str:
    return Path(str(name)).name.split("_", 1)[0]


def _load_speaker_npz(speaker: str) -> tuple[list[str], list[np.ndarray]] | None:
    path = WHISPER_EMBEDDING_DIR / f"{speaker}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    fnames = [str(x) for x in data["filenames"]]
    frames = [data[f"frames_{i}"] for i in range(len(fnames))]
    return fnames, frames


def _gather_train_control_frames(
    train_filenames: Iterable[str],
    max_control_frames: int,
) -> tuple[np.ndarray, int, int]:




    train_set = {str(f) for f in train_filenames}
    chunks: list[np.ndarray] = []
    n_utts = 0
    for spk in CONTROL_SPEAKERS:
        loaded = _load_speaker_npz(spk)
        if loaded is None:
            print(f"  [warn] no Whisper cache for control speaker {spk}; skipped")
            continue
        fnames, frames_list = loaded
        for fname, frames in zip(fnames, frames_list):
            if fname in train_set:
                chunks.append(frames)
                n_utts += 1
    if not chunks:
        raise RuntimeError(
            "No train-side control utterances found in Whisper cache "
            f"({WHISPER_EMBEDDING_DIR}). Did you build the encoder cache first?"
        )

    ctrl_frames = np.concatenate(chunks, axis=0)
    n_pre = ctrl_frames.shape[0]
    if ctrl_frames.shape[0] > max_control_frames:
        rng = np.random.RandomState(FDS_SEED)
        keep = rng.choice(ctrl_frames.shape[0], max_control_frames, replace=False)
        ctrl_frames = ctrl_frames[keep]
        print(
            f"  Subsampled train-control frames: {n_pre} -> {ctrl_frames.shape[0]} "
            f"(seed={FDS_SEED})"
        )
    return ctrl_frames, n_utts, n_pre


def _score_all_speakers(model: FrameLevelFDS) -> dict[str, np.ndarray]:

    mapping: dict[str, np.ndarray] = {}
    all_speakers = CONTROL_SPEAKERS + list(DYSARTHRIC_SPEAKERS.keys())
    for si, spk in enumerate(all_speakers):
        loaded = _load_speaker_npz(spk)
        if loaded is None:
            continue
        fnames, frames_list = loaded
        if not fnames:
            continue
        all_frames = np.concatenate(frames_list, axis=0)
        t_spk = time.time()
        all_scores = model.score_frames(all_frames).astype(np.float32)
        offset = 0
        for fname, frames in zip(fnames, frames_list):
            n = len(frames)
            mapping[str(fname)] = all_scores[offset:offset + n]
            offset += n
        elapsed = time.time() - t_spk
        print(
            f"  [{si+1}/{len(all_speakers)}] {spk}: {len(fnames)} utts, "
            f"{all_frames.shape[0]} frames, {elapsed:.1f}s"
        )
    return mapping


def _save_raw_scores(path: Path, mapping: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fnames = sorted(mapping)
    payload = {"filenames": np.asarray(fnames)}
    for idx, fname in enumerate(fnames):
        payload[f"scores_{idx}"] = mapping[fname].astype(np.float32)
    np.savez_compressed(str(path), **payload)


def _train_control_zscore_stats(
    raw_map: dict[str, np.ndarray],
    train_filenames: Iterable[str],
) -> tuple[float, float]:
    train_set = {str(f) for f in train_filenames}
    chunks: list[np.ndarray] = []
    for fname, scores in raw_map.items():
        if _filename_to_speaker(fname) in CONTROL_SPEAKERS and fname in train_set:
            chunks.append(scores)
    if not chunks:
        return 0.0, 1.0
    values = np.concatenate(chunks).astype(np.float32)
    mean = float(values.mean())
    std = float(values.std() + 1e-6)
    return mean, std


def _outputs_complete(out_dir: Path) -> bool:
    return all(
        (out_dir / name).exists()
        for name in (
            "frame_fds_model.npz",
            "raw_frame_scores.npz",
            "zscore_stats.json",
            "manifest.json",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-seed train-control-only frame-level FDS fitter for TORGO."
    )
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--K", type=int, default=FDS_K)
    parser.add_argument("--T", type=int, default=FDS_T)
    parser.add_argument("--max-control-frames", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fit even if all output files already exist.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if _outputs_complete(out_dir) and not args.force:
        print(f"[skip] Train-only FDS cache already complete: {out_dir}")
        return

    split = _load_split(args.split_file)
    train_files = split["train"]
    train_set = {str(f) for f in train_files}
    train_ctrl = sorted(
        f for f in train_set if _filename_to_speaker(f) in CONTROL_SPEAKERS
    )

    print(f"[fit] split={args.split_file}")
    print(f"[fit] |train|={len(train_files)}, |train_control|={len(train_ctrl)}")
    print(f"[fit] Whisper cache: {WHISPER_EMBEDDING_DIR}")

    ctrl_frames, n_train_ctrl_utts, n_frames_pre = _gather_train_control_frames(
        train_ctrl, max_control_frames=args.max_control_frames
    )
    print(
        f"[fit] train-control frames: {ctrl_frames.shape[0]} (pre-subsample={n_frames_pre}, "
        f"utts={n_train_ctrl_utts}, dim={ctrl_frames.shape[1]})"
    )

    model = FrameLevelFDS(K=args.K, T=args.T, device=args.device)
    t0 = time.time()
    model.fit(ctrl_frames)
    fit_secs = time.time() - t0
    print(f"[fit] FrameLevelFDS fit done in {fit_secs:.1f}s")
    del ctrl_frames

    model_path = out_dir / "frame_fds_model.npz"
    model.save(model_path)
    print(f"[fit] saved fitted model -> {model_path}")

    print("[score] scoring all utterances against train-only reference ...")
    raw_map = _score_all_speakers(model)
    raw_path = out_dir / "raw_frame_scores.npz"
    _save_raw_scores(raw_path, raw_map)
    print(f"[score] wrote per-utterance frame scores -> {raw_path} ({len(raw_map)} utts)")

    zscore_mean, zscore_std = _train_control_zscore_stats(raw_map, train_set)
    zscore_path = out_dir / "zscore_stats.json"
    zscore_path.write_text(
        json.dumps({"mean": zscore_mean, "std": zscore_std}, indent=2),
        encoding="utf-8",
    )
    print(
        f"[score] z-score stats (train-control only): "
        f"mean={zscore_mean:.4f}, std={zscore_std:.4f} -> {zscore_path}"
    )

    manifest = {
        "kind": "train_only_fds",
        "split_file": str(args.split_file),
        "split_seed": split.get("seed"),
        "split_protocol": split.get("protocol"),
        "n_train_total": len(train_files),
        "n_train_control_utts": n_train_ctrl_utts,
        "n_train_control_frames_pre_subsample": int(n_frames_pre),
        "max_control_frames": int(args.max_control_frames),
        "control_speakers": list(CONTROL_SPEAKERS),
        "K": int(args.K),
        "T": int(args.T),
        "fds_seed": int(FDS_SEED),
        "device": str(args.device),
        "whisper_cache": str(WHISPER_EMBEDDING_DIR),
        "fit_seconds": float(fit_secs),
        "n_utterances_scored": len(raw_map),
        "zscore_mean": zscore_mean,
        "zscore_std": zscore_std,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
