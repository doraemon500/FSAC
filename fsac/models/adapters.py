from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.act = nn.GELU()

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gate) * self.up(self.act(self.down(h)))


def _masked_softmax(
    attn_scores: torch.Tensor,
    source_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if source_mask is None:
        return F.softmax(attn_scores, dim=-1)

    mask = source_mask.unsqueeze(1).unsqueeze(2).bool()
    valid = mask.any(dim=-1, keepdim=True)
    masked = attn_scores.masked_fill(~mask, float("-inf"))
    masked = torch.where(valid, masked, torch.zeros_like(masked))
    weights = F.softmax(masked, dim=-1)
    return torch.where(valid, weights, torch.zeros_like(weights))


class PlainAcousticAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_whisper: int,
        bottleneck: int = 64,
        n_heads: int = 4,
        gate_init: float | None = 0.0,
    ):
        super().__init__()
        if bottleneck % n_heads != 0:
            raise ValueError("bottleneck must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = bottleneck // n_heads
        self.use_global_gate = gate_init is not None

        self.q_proj = nn.Linear(d_model, bottleneck)
        self.k_proj = nn.Linear(d_whisper, bottleneck)
        self.v_proj = nn.Linear(d_whisper, bottleneck)
        self.out_proj = nn.Linear(bottleneck, d_model)

        if self.use_global_gate:
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_buffer("gate", None)

        self.layer_norm = nn.LayerNorm(d_model)

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _apply_global_gate(self, out: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gate) * out

    def forward(
        self,
        h: torch.Tensor,
        whisper_states: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = h.shape
        n_frames = whisper_states.shape[1]

        h_norm = self.layer_norm(h)
        q = self.q_proj(h_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(whisper_states).view(bsz, n_frames, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(whisper_states).view(bsz, n_frames, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = _masked_softmax(attn_scores, source_mask)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.out_proj(out)

        if self.use_global_gate:
            out = self._apply_global_gate(out)
        return out


class VGateAcousticAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_whisper: int,
        bottleneck: int = 64,
        n_heads: int = 4,
        gate_arch: str = "linear",
        gate_init: float | None = 0.0,
        attn_risk_mode: str = "v_only",
        risk_scale: str = "raw",
    ):
        super().__init__()
        if bottleneck % n_heads != 0:
            raise ValueError("bottleneck must be divisible by n_heads")
        if attn_risk_mode != "v_only":
            raise ValueError("FSAC supports only attn_risk_mode='v_only'")

        self.n_heads = n_heads
        self.head_dim = bottleneck // n_heads
        self.attn_risk_mode = attn_risk_mode
        self.risk_scale = risk_scale
        self.use_global_gate = gate_init is not None

        self.q_proj = nn.Linear(d_model, bottleneck)
        self.k_proj = nn.Linear(d_whisper, bottleneck)
        self.v_proj = nn.Linear(d_whisper, bottleneck)
        self.out_proj = nn.Linear(bottleneck, d_model)

        if self.use_global_gate:
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_buffer("gate", None)

        self.layer_norm = nn.LayerNorm(d_model)

        if risk_scale == "learnable":
            self.risk_alpha = nn.Parameter(torch.tensor(5.0))

        if gate_arch == "mlp":
            self.risk_gate = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, self.head_dim),
                nn.Sigmoid(),
            )
        elif gate_arch == "linear":
            self.risk_gate = nn.Sequential(nn.Linear(1, self.head_dim), nn.Sigmoid())
        else:
            raise ValueError(f"Unknown gate_arch: {gate_arch}")

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        whisper_states: torch.Tensor,
        risk_scores: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = h.shape
        n_frames = whisper_states.shape[1]

        h_norm = self.layer_norm(h)
        q = self.q_proj(h_norm).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(whisper_states).view(bsz, n_frames, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(whisper_states).view(bsz, n_frames, self.n_heads, self.head_dim).transpose(1, 2)

        if risk_scores is not None and self.risk_scale == "learnable":
            risk_scores = self.risk_alpha * risk_scores

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = _masked_softmax(attn_scores, source_mask)

        if risk_scores is not None:
            gate = self.risk_gate(risk_scores.unsqueeze(-1)).unsqueeze(1)
            v = v * gate

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.out_proj(out)

        if self.use_global_gate:
            out = self._apply_global_gate(out)
        return out

    def _apply_global_gate(self, out: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gate) * out


def _build_acoustic_adapter(
    adapter_mode: str,
    d_model: int,
    d_whisper: int,
    bottleneck: int,
    n_heads: int,
    gate_arch: str,
    gate_init: float | None,
    attn_risk_mode: str,
    risk_scale: str,
):
    if adapter_mode == "plain":
        return PlainAcousticAdapter(
            d_model=d_model,
            d_whisper=d_whisper,
            bottleneck=bottleneck,
            n_heads=n_heads,
            gate_init=gate_init,
        )
    if adapter_mode == "v_gate":
        return VGateAcousticAdapter(
            d_model=d_model,
            d_whisper=d_whisper,
            bottleneck=bottleneck,
            n_heads=n_heads,
            gate_arch=gate_arch,
            gate_init=gate_init,
            attn_risk_mode=attn_risk_mode,
            risk_scale=risk_scale,
        )
    raise ValueError(f"Unsupported FSAC adapter mode: {adapter_mode}")


class WhisperingLLMLayerWrapper(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        d_model: int,
        d_whisper: int,
        bottleneck: int = 64,
        n_heads: int = 4,
        gate_arch: str = "linear",
        gate_init: float | None = 0.0,
        attn_risk_mode: str = "v_only",
        risk_scale: str = "raw",
        adapter_mode: str = "v_gate",
        severity_strategy: str = "adaptive_threshold",
    ):
        super().__init__()
        self.base_layer = base_layer
        self.adapter_mode = adapter_mode
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.adapter_L = BottleneckAdapter(d_model, bottleneck)
        self.adapter_W = _build_acoustic_adapter(
            adapter_mode=adapter_mode,
            d_model=d_model,
            d_whisper=d_whisper,
            bottleneck=bottleneck,
            n_heads=n_heads,
            gate_arch=gate_arch,
            gate_init=gate_init,
            attn_risk_mode=attn_risk_mode,
            risk_scale=risk_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        whisper_states: torch.Tensor | None = None,
        risk_scores: torch.Tensor | None = None,
        whisper_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        layer_out = self.base_layer(hidden_states, **kwargs)
        if isinstance(layer_out, tuple):
            h = layer_out[0]
            rest = layer_out[1:]
        else:
            h = layer_out
            rest = ()

        h = h + self.adapter_L(h)
        if whisper_states is not None:
            if self.adapter_mode == "v_gate":
                h = h + self.adapter_W(
                    h,
                    whisper_states,
                    risk_scores=risk_scores,
                    source_mask=whisper_mask,
                )
            else:
                h = h + self.adapter_W(h, whisper_states, source_mask=whisper_mask)

        if rest:
            return (h,) + rest
        return h
