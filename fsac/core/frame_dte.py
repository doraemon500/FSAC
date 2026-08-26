"""
Frame-level DTE scoring using encoder frame features.

https://arxiv.org/abs/2305.18593

Applies the full DTE-NP formulation (inverse-gamma kernel -> diffusion
timestep estimation) at frame granularity using FAISS for fast kNN.

The flow of the code baseline:
  1. kNN distance per frame
  2. inverse-gamma posterior over T diffusion timesteps
  3. argmax -> estimated diffusion timestep (= anomaly score)

This module is dataset-agnostic. Dataset-specific CLI wrappers live in
fsac/datasets/<dataset>/frame_dte.py.
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler


class FrameLevelDTE:
    def __init__(self, K=5, T=1000, device="cuda"):
        self.K = K
        self.T = T
        self.device = device
        self.index = None
        self.scaler = None
        self.dim = None
        self._scaled_ref = None

        beta_0, beta_T = 0.0001, 0.01
        betas = np.linspace(beta_0, beta_T, T)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.one_minus_ac = (1.0 - alphas_cumprod).astype(np.float64)
        self.log_one_minus_ac = np.log(self.one_minus_ac)
        self.inv_one_minus_ac = 1.0 / self.one_minus_ac

    def fit(self, control_frames: np.ndarray):
        self.dim = control_frames.shape[1]
        self.scaler = StandardScaler().fit(control_frames)
        scaled = self.scaler.transform(control_frames).astype(np.float32)
        self._scaled_ref = scaled

        self._use_torch_knn = False

        if self.device == "cuda":
            try:
                import faiss
                if faiss.get_num_gpus() > 0:
                    index_flat = faiss.IndexFlatL2(self.dim)
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                    self.index.add(scaled)
                    print("    [FAISS] GPU index active")
                    return
                else:
                    print("    [FAISS] faiss.get_num_gpus()=0, falling back to PyTorch CUDA kNN")
            except ImportError:
                print("    [FAISS] not installed, falling back to PyTorch CUDA kNN")

            import torch
            if torch.cuda.is_available():
                self._use_torch_knn = True
                self._ref_gpu = torch.from_numpy(scaled).cuda()
                self._ref_norms = (self._ref_gpu ** 2).sum(dim=1, keepdim=True)
                print(f"    [PyTorch] CUDA kNN active (ref={scaled.shape[0]:,} x {scaled.shape[1]})")
                return

        try:
            import faiss
            index_flat = faiss.IndexFlatL2(self.dim)
            faiss.omp_set_num_threads(min(32, __import__('os').cpu_count() or 4))
            self.index = index_flat
            self.index.add(scaled)
            print(f"    [FAISS] CPU index (threads={faiss.omp_get_max_threads()})")
        except ImportError:
            raise RuntimeError("Neither faiss nor torch CUDA available for kNN")

    def _knn_distances(self, frames: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(frames).astype(np.float32)

        if self._use_torch_knn:
            return self._knn_distances_torch(scaled)

        dists_sq, _ = self.index.search(scaled, self.K)
        return np.sqrt(np.clip(dists_sq, 0, None)).mean(axis=1)

    def _knn_distances_torch(self, scaled: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        import torch
        N = scaled.shape[0]
        result = np.empty(N, dtype=np.float32)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            q = torch.from_numpy(scaled[start:end]).cuda()
            q_norms = (q ** 2).sum(dim=1, keepdim=True)
            dists_sq = q_norms + self._ref_norms.T - 2.0 * (q @ self._ref_gpu.T)
            dists_sq.clamp_(min=0)
            topk_dists, _ = dists_sq.topk(self.K, dim=1, largest=False)
            mean_dist = topk_dists.sqrt().mean(dim=1)
            result[start:end] = mean_dist.cpu().numpy()
            del q, q_norms, dists_sq, topk_dists

        return result

    def _dte_scores(self, mean_dists: np.ndarray, batch_size: int = 50000) -> np.ndarray:
        N = len(mean_dists)
        scores = np.empty(N, dtype=np.float32)
        a = 0.5 * self.dim - 1

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            s = (mean_dists[start:end, None] / 2).astype(np.float64)

            log_density = (
                a * np.log(s)
                - (a + 1) * self.log_one_minus_ac[None, :]
                - s * self.inv_one_minus_ac[None, :]
            )

            log_density -= logsumexp(log_density, axis=1, keepdims=True)
            scores[start:end] = np.argmax(log_density, axis=1).astype(np.float32)

        return scores

    def score_frames(self, frames: np.ndarray) -> np.ndarray:
        mean_dists = self._knn_distances(frames)
        return self._dte_scores(mean_dists)

    def save(self, path) -> None:
        if self._scaled_ref is None or self.scaler is None:
            raise RuntimeError("save() called before fit().")
        np.savez_compressed(
            str(path),
            K=np.array(self.K),
            T=np.array(self.T),
            dim=np.array(self.dim),
            scaler_mean=self.scaler.mean_.astype(np.float32),
            scaler_scale=self.scaler.scale_.astype(np.float32),
            scaled_ref=self._scaled_ref,
        )

    def load(self, path) -> None:
        data = np.load(str(path), allow_pickle=False)
        self.K = int(data["K"])
        self.T = int(data["T"])
        self.dim = int(data["dim"])

        self.scaler = StandardScaler()
        self.scaler.mean_ = data["scaler_mean"].astype(np.float64)
        self.scaler.scale_ = data["scaler_scale"].astype(np.float64)
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = self.dim
        self.scaler.n_samples_seen_ = 1

        scaled = data["scaled_ref"]
        self._scaled_ref = scaled
        self._use_torch_knn = False

        try:
            import faiss
            index_flat = faiss.IndexFlatL2(self.dim)
            faiss.omp_set_num_threads(min(32, __import__('os').cpu_count() or 4))
            self.index = index_flat
            self.index.add(scaled)
        except ImportError:
            raise RuntimeError("faiss is required to restore a cached FrameLevelDTE model.")

    def score_words(self, frames, word_boundaries):
        frame_risks = self.score_frames(frames)
        results = []
        for s, e, w in word_boundaries:
            seg = frame_risks[s:e]
            if len(seg) == 0:
                results.append((w, 0.0, 0.0))
            else:
                results.append((w, float(seg.mean()), float(seg.max())))
        return results
