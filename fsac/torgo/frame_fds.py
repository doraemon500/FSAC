import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fsac.core.frame_fds import FrameLevelFDS
from fsac.torgo.config import (
    CONTROL_SPEAKERS,
    FDS_K,
    FDS_T,
    DYSARTHRIC_SPEAKERS,
    SPEAKER_TO_SEVERITY,
    WHISPER_FDS_RESULT_DIR,
    WHISPER_EMBEDDING_DIR,
)


def load_frames(speaker_id: str):
    path = WHISPER_EMBEDDING_DIR / f"{speaker_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    fnames = data["filenames"]
    frames_list = [data[f"frames_{i}"] for i in range(len(fnames))]
    return fnames, frames_list


def main():
    parser = argparse.ArgumentParser(description="Frame-level FDS-NP for TORGO")
    parser.add_argument("--K", type=int, default=FDS_K)
    parser.add_argument("--T", type=int, default=FDS_T)
    parser.add_argument("--max-control-frames", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    WHISPER_FDS_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting control speaker frames ...")
    all_ctrl_frames = []
    for spk in CONTROL_SPEAKERS:
        result = load_frames(spk)
        if result is None:
            continue
        for frames in result[1]:
            all_ctrl_frames.append(frames)

    ctrl_frames = np.concatenate(all_ctrl_frames, axis=0)
    print(f"  Total control frames: {ctrl_frames.shape[0]}, dim={ctrl_frames.shape[1]}")

    if ctrl_frames.shape[0] > args.max_control_frames:
        rng = np.random.RandomState(42)
        idx = rng.choice(ctrl_frames.shape[0], args.max_control_frames, replace=False)
        ctrl_frames = ctrl_frames[idx]
        print(f"  Subsampled to {ctrl_frames.shape[0]} frames")

    print(f"Building kNN index (K={args.K}, T={args.T}, device={args.device}) ...")
    t0 = time.time()
    fds = FrameLevelFDS(K=args.K, T=args.T, device=args.device)
    fds.fit(ctrl_frames)
    print(f"  Done in {time.time() - t0:.1f}s")
    del ctrl_frames

    cache_dir = WHISPER_EMBEDDING_DIR / "fds_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fds_model_cache = cache_dir / "frame_fds_model.npz"
    fds.save(fds_model_cache)
    print(f"  Saved fitted FDS model cache to {fds_model_cache}")

    print("Scoring all utterances ...")
    rows = []
    raw_score_map = {}
    control_score_arrays = []
    all_speakers = CONTROL_SPEAKERS + list(DYSARTHRIC_SPEAKERS.keys())

    for si, spk in enumerate(all_speakers):
        result = load_frames(spk)
        if result is None:
            continue

        fnames, frames_list = result
        is_ctrl = spk in CONTROL_SPEAKERS
        severity = None if is_ctrl else SPEAKER_TO_SEVERITY.get(spk)

        t_spk = time.time()
        all_frames = np.concatenate(frames_list, axis=0)
        all_scores = fds.score_frames(all_frames)

        offset = 0
        for i, fname in enumerate(fnames):
            n = len(frames_list[i])
            utt_scores = all_scores[offset:offset + n]
            offset += n
            raw_score_map[str(fname)] = utt_scores.astype(np.float32)
            if is_ctrl:
                control_score_arrays.append(utt_scores.astype(np.float32))
            rows.append({
                "speaker": spk,
                "filename": str(fname),
                "is_control": is_ctrl,
                "severity": severity,
                "frame_fds_mean": float(utt_scores.mean()),
                "frame_fds_max": float(utt_scores.max()),
                "frame_fds_std": float(utt_scores.std()),
                "n_frames": n,
            })
        elapsed = time.time() - t_spk
        print(f"  [{si+1}/{len(all_speakers)}] {spk}: {len(fnames)} utts, "
              f"{all_frames.shape[0]} frames, {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    out_csv = WHISPER_FDS_RESULT_DIR / "frame_fds_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    raw_cache = cache_dir / "raw_frame_scores.npz"
    fnames = sorted(raw_score_map)
    raw_payload = {"filenames": np.asarray(fnames)}
    for idx, fname in enumerate(fnames):
        raw_payload[f"scores_{idx}"] = raw_score_map[fname]
    np.savez_compressed(str(raw_cache), **raw_payload)
    print(f"Saved per-frame FDS cache to {raw_cache}")

    if control_score_arrays:
        ctrl_scores_all = np.concatenate(control_score_arrays).astype(np.float32)
        stats = {
            "mean": float(ctrl_scores_all.mean()),
            "std": float(ctrl_scores_all.std() + 1e-6),
        }
        stats_path = cache_dir / "zscore_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Saved FDS z-score stats to {stats_path}")

    ctrl = df[df["is_control"]]["frame_fds_mean"]
    dys = df[~df["is_control"]]["frame_fds_mean"]
    print(f"  Control:    mean={ctrl.mean():.4f}  std={ctrl.std():.4f}")
    print(f"  Dysarthric: mean={dys.mean():.4f}  std={dys.std():.4f}")


if __name__ == "__main__":
    main()
