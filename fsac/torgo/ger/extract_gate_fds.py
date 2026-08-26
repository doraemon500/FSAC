from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fsac.models.adapters import VGateAcousticAdapter
from fsac.torgo.config import FDS_T


def _scale_risk(
    raw: np.ndarray,
    mode: str,
    z_mean: float,
    z_std: float,
    scale_param: float | None,
) -> np.ndarray:

    if mode == "raw":
        return raw / FDS_T
    if mode == "zscore":
        return (raw - z_mean) / max(z_std, 1e-6)
    if mode == "ztemp":
        z = (raw - z_mean) / max(z_std, 1e-6)
        return z * float(scale_param if scale_param is not None else 1.0)
    if mode == "temp":
        return (raw / FDS_T) * float(scale_param if scale_param is not None else 1.0)
    return raw / FDS_T


def _parse_risk_scale(spec: str) -> tuple[str, float | None]:
    if spec is None:
        return "raw", None
    if ":" in str(spec):
        mode, param = str(spec).split(":", 1)
        try:
            return mode, float(param)
        except ValueError:
            return mode, None
    return str(spec), None


def _load_split(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_raw_fds_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    fnames = [str(x) for x in data["filenames"]]
    return {fname: data[f"scores_{idx}"].astype(np.float32) for idx, fname in enumerate(fnames)}


def _build_adapter(gating_cfg: dict, d_whisper: int, bottleneck: int, n_heads: int) -> VGateAcousticAdapter:
    return VGateAcousticAdapter(
        d_model=1,
        d_whisper=d_whisper,
        bottleneck=bottleneck,
        n_heads=n_heads,
        gate_arch=gating_cfg.get("gate_arch", "linear"),
        gate_init=(
            None if gating_cfg.get("gate_init") in (None, "none")
            else float(gating_cfg.get("gate_init"))
        ),
        attn_risk_mode=gating_cfg.get("attn_risk_mode", "v_only"),
        risk_scale=gating_cfg.get("risk_scale", "raw"),
    )


def _load_risk_gate_modules(
    ckpt: dict, gating_cfg: dict, d_whisper: int, bottleneck: int, n_heads: int,
) -> dict[str, VGateAcousticAdapter]:

    state = ckpt.get("wrapped_layers_trainable", {})
    if not state:
        raise KeyError(
            "Checkpoint has no 'wrapped_layers_trainable' state; cannot extract gate."
        )
    modules: dict[str, VGateAcousticAdapter] = {}
    for layer_idx, layer_state in state.items():
        adapter_state = layer_state.get("adapter_W", {})
        risk_gate_state = {
            k.removeprefix("risk_gate."): v
            for k, v in adapter_state.items()
            if k.startswith("risk_gate.")
        }
        if not risk_gate_state:
            continue
        adapter = _build_adapter(gating_cfg, d_whisper, bottleneck, n_heads)
        missing, unexpected = adapter.risk_gate.load_state_dict(risk_gate_state, strict=False)
        if any("risk_gate" in m for m in missing):
            raise KeyError(
                f"Layer {layer_idx}: incompatible risk_gate state "
                f"(missing={missing}, unexpected={unexpected})"
            )
        modules[layer_idx] = adapter
    if not modules:
        raise RuntimeError("No risk_gate modules could be loaded from checkpoint.")
    return modules


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    sx = float(x.std())
    sy = float(y.std())
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-cell gate-FDS correlation extractor.")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Cell run dir, e.g. results/.../largev2/qwen3_8b/seed42")
    parser.add_argument("--suffix", required=True, type=str,
                        help="Trained model directory suffix, e.g. _vgate_s42")
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--fds-cache-dir", required=True, type=Path,
                        help="Per-seed FDS cache (train-only) used at training time.")
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d-whisper", type=int, default=1280, help="Default for large-v2.")
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cell_dir = args.run_dir / f"whispering_llm{args.suffix}"
    ckpt_path = args.checkpoint or (cell_dir / "best_checkpoint.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    out_pairs = cell_dir / "gate_fds_pairs.npz"
    out_corr = cell_dir / "gate_fds_corr.json"
    if out_pairs.exists() and out_corr.exists() and not args.force:
        print(f"[skip] gate-FDS pairs already extracted at {cell_dir}")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[load] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gating_cfg = ckpt.get("gating_config", {})
    risk_zscore_stats = ckpt.get("risk_zscore_stats")
    risk_scale_mode, risk_scale_param = _parse_risk_scale(gating_cfg.get("risk_scale", "raw"))

    z_mean, z_std = (0.0, 1.0)
    if risk_zscore_stats:
        z_mean, z_std = float(risk_zscore_stats[0]), float(risk_zscore_stats[1])
    else:
        zfile = args.fds_cache_dir / "zscore_stats.json"
        if zfile.exists():
            payload = json.loads(zfile.read_text(encoding="utf-8"))
            z_mean = float(payload["mean"])
            z_std = float(payload["std"])
    print(f"[load] risk_scale={risk_scale_mode}, z_mean={z_mean:.4f}, z_std={z_std:.4f}")

    print(f"[load] FDS cache: {args.fds_cache_dir}")
    raw_path = args.fds_cache_dir / "raw_frame_scores.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"Per-seed raw frame scores not found: {raw_path}")
    raw_scores = _load_raw_fds_cache(raw_path)

    split = _load_split(args.split_file)
    test_files = [str(f) for f in split.get("test", [])]
    if not test_files:
        raise RuntimeError(f"Split file has no 'test' key: {args.split_file}")
    print(f"[load] split test set: {len(test_files)} utterances")

    modules = _load_risk_gate_modules(
        ckpt, gating_cfg, args.d_whisper, args.bottleneck, args.n_heads,
    )
    for mod in modules.values():
        mod.eval()
        mod.to(device)
    n_layers = len(modules)
    print(f"[load] {n_layers} wrapped layers with risk_gate weights")

    per_utt = {}
    skipped = 0
    for fname in test_files:
        if fname not in raw_scores:
            skipped += 1
            continue
        raw = raw_scores[fname]
        if raw.size == 0:
            skipped += 1
            continue
        scaled = _scale_risk(raw.astype(np.float32), risk_scale_mode, z_mean, z_std, risk_scale_param)
        scaled_t = torch.from_numpy(scaled).float().to(device).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            gate_norms = []
            for mod in modules.values():
                gate = mod.risk_gate(scaled_t)
                gate_norms.append(gate.norm(dim=-1).squeeze(0))
            avg_gate_norm = torch.stack(gate_norms, dim=0).mean(dim=0)
        per_utt[fname] = {
            "gate_norm": avg_gate_norm.detach().cpu().numpy().astype(np.float32),
            "fds": raw.astype(np.float32),
        }

    if not per_utt:
        raise RuntimeError("No test utterances had per-frame FDS scores in the FDS cache.")

    g_all = np.concatenate([v["gate_norm"] for v in per_utt.values()])
    f_all = np.concatenate([v["fds"] for v in per_utt.values()])
    pearson = _pearson_r(g_all, f_all)
    spearman = _spearman_rho(g_all, f_all)

    payload = {"filenames": np.asarray(sorted(per_utt))}
    for idx, fname in enumerate(sorted(per_utt)):
        payload[f"gate_norm_{idx}"] = per_utt[fname]["gate_norm"]
        payload[f"fds_{idx}"] = per_utt[fname]["fds"]
    np.savez_compressed(str(out_pairs), **payload)
    print(f"[save] per-utt pairs -> {out_pairs}  ({len(per_utt)} utts, {g_all.size} frames)")

    summary = {
        "checkpoint": str(ckpt_path),
        "fds_cache_dir": str(args.fds_cache_dir),
        "risk_scale_mode": risk_scale_mode,
        "z_mean": z_mean,
        "z_std": z_std,
        "n_test_utts_total": len(test_files),
        "n_test_utts_used": len(per_utt),
        "n_test_utts_skipped": skipped,
        "n_frames_total": int(g_all.size),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "gate_norm_mean": float(g_all.mean()),
        "gate_norm_std": float(g_all.std()),
        "fds_mean": float(f_all.mean()),
        "fds_std": float(f_all.std()),
        "n_wrapped_layers": int(n_layers),
        "acoustic_fds_reference": ckpt.get("acoustic_fds_reference", "all_control"),
    }
    out_corr.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[save] correlation -> {out_corr}  "
        f"pearson={pearson:.4f}  spearman={spearman:.4f}  n_frames={g_all.size}"
    )


if __name__ == "__main__":
    main()
