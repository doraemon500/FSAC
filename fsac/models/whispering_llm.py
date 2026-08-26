import torch
import torch.nn as nn

from fsac.models.adapters import WhisperingLLMLayerWrapper


def _to_cpu_state(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_state(v) for v in obj)
    return obj


class WhisperProjection(nn.Module):
    def __init__(self, d_whisper: int, d_target: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_whisper, d_target),
            nn.GELU(),
            nn.LayerNorm(d_target),
        )

    def forward(self, whisper_states: torch.Tensor) -> torch.Tensor:
        return self.proj(whisper_states)


class FSACWhisperingLLM(nn.Module):
    def __init__(
        self,
        base_lm,
        d_whisper: int = 512,
        adapter_bottleneck: int = 64,
        adapter_n_heads: int = 4,
        wrap_every_n: int = 1,
        gate_arch: str = "linear",
        gate_init: float = 0.0,
        attn_risk_mode: str = "v_only",
        risk_scale: str = "raw",
        adapter_mode: str = "v_gate",
        severity_strategy: str = "adaptive_threshold",
    ):
        super().__init__()
        self.lm = base_lm
        self.d_model = base_lm.config.hidden_size

        self.gate_arch = gate_arch
        self.gate_init = gate_init
        self.attn_risk_mode = attn_risk_mode
        self.risk_scale = risk_scale
        self.adapter_mode = adapter_mode
        self.severity_strategy = severity_strategy

        for param in self.lm.parameters():
            param.requires_grad = False

        self.whisper_proj = WhisperProjection(d_whisper, d_whisper)

        self._param_dtype = next(self.lm.parameters()).dtype

        self.wrapped_layers = nn.ModuleDict()
        layers = self.lm.model.layers
        for i in range(len(layers)):
            if i % wrap_every_n == 0:
                wrapper = WhisperingLLMLayerWrapper(
                    base_layer=layers[i],
                    d_model=self.d_model,
                    d_whisper=d_whisper,
                    bottleneck=adapter_bottleneck,
                    n_heads=adapter_n_heads,
                    gate_arch=gate_arch,
                    gate_init=gate_init,
                    attn_risk_mode=attn_risk_mode,
                    risk_scale=risk_scale,
                    adapter_mode=adapter_mode,
                    severity_strategy=severity_strategy,
                )
                self.wrapped_layers[str(i)] = wrapper

        self.whisper_proj = self.whisper_proj.to(dtype=self._param_dtype)
        for wrapper in self.wrapped_layers.values():
            wrapper.adapter_L = wrapper.adapter_L.to(dtype=self._param_dtype)
            wrapper.adapter_W = wrapper.adapter_W.to(dtype=self._param_dtype)

    def get_trainable_parameters(self):
        params = []
        params.extend(self.whisper_proj.parameters())
        for wrapper in self.wrapped_layers.values():
            params.extend(wrapper.adapter_L.parameters())
            params.extend(wrapper.adapter_W.parameters())
        return params

    def count_trainable_params(self):
        total = sum(p.numel() for p in self.get_trainable_parameters())
        all_params = sum(p.numel() for p in self.parameters())
        return total, all_params

    def get_checkpoint_state(self):
        return {
            "checkpoint_format": "trainable_only_v1",
            "whisper_proj": _to_cpu_state(self.whisper_proj.state_dict()),
            "wrapped_layers_trainable": {
                k: {
                    "adapter_L": _to_cpu_state(v.adapter_L.state_dict()),
                    "adapter_W": _to_cpu_state(v.adapter_W.state_dict()),
                }
                for k, v in self.wrapped_layers.items()
            },
        }

    def load_checkpoint_state(self, ckpt: dict):
        self.whisper_proj.load_state_dict(ckpt["whisper_proj"])
        if "wrapped_layers_trainable" in ckpt:
            for k, layer_sd in ckpt["wrapped_layers_trainable"].items():
                if k not in self.wrapped_layers:
                    continue
                if "adapter_L" in layer_sd:
                    self.wrapped_layers[k].adapter_L.load_state_dict(layer_sd["adapter_L"])
                if "adapter_W" in layer_sd:
                    self.wrapped_layers[k].adapter_W.load_state_dict(layer_sd["adapter_W"])
            return
        if "wrapped_layers" in ckpt:
            for k, v_sd in ckpt["wrapped_layers"].items():
                if k in self.wrapped_layers:
                    self.wrapped_layers[k].load_state_dict(v_sd)
            return
        raise KeyError("Checkpoint missing wrapped layer state")

    def get_adapter_diagnostics(self):
        return {}

    def _cast_inputs(self, whisper_frames, frame_risks):
        target_dtype = next(self.whisper_proj.parameters()).dtype
        if whisper_frames is not None:
            whisper_frames = whisper_frames.to(dtype=target_dtype)
        if frame_risks is not None:
            frame_risks = frame_risks.to(dtype=target_dtype)
        return whisper_frames, frame_risks

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        whisper_frames: torch.Tensor = None,
        frame_risks: torch.Tensor = None,
        whisper_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        whisper_frames, frame_risks = self._cast_inputs(whisper_frames, frame_risks)

        whisper_states = None
        risk_scores = None

        if whisper_frames is not None:
            whisper_states = self.whisper_proj(whisper_frames)
            risk_scores = frame_risks

        embed_tokens = self.lm.model.embed_tokens
        hidden_states = embed_tokens(input_ids)

        if hasattr(self.lm.model, 'norm'):
            final_norm = self.lm.model.norm
        else:
            final_norm = None

        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        if hasattr(self.lm.model, 'rotary_emb'):
            position_embeddings = self.lm.model.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        layers = self.lm.model.layers
        for i, layer in enumerate(layers):
            extra_kwargs = {}
            if position_embeddings is not None:
                extra_kwargs["position_embeddings"] = position_embeddings
            if str(i) in self.wrapped_layers:
                wrapper = self.wrapped_layers[str(i)]
                result = wrapper(
                    hidden_states,
                    whisper_states=whisper_states,
                    risk_scores=risk_scores,
                    whisper_mask=whisper_mask,
                    **extra_kwargs,
                )
                if isinstance(result, tuple):
                    hidden_states = result[0]
                else:
                    hidden_states = result
            else:
                result = layer(hidden_states, **extra_kwargs)
                if isinstance(result, tuple):
                    hidden_states = result[0]
                else:
                    hidden_states = result

        if final_norm is not None:
            hidden_states = final_norm(hidden_states)

        logits = self.lm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels.view(-1) != -100
            if not valid_mask.any():
                loss = shift_logits.sum() * 0.0
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = per_token_loss[valid_mask].mean()

        outputs = {"loss": loss, "logits": logits}
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        whisper_frames: torch.Tensor = None,
        frame_risks: torch.Tensor = None,
        whisper_mask: torch.Tensor = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ):
        whisper_frames, frame_risks = self._cast_inputs(whisper_frames, frame_risks)

        whisper_states = None
        risk_scores = None

        if whisper_frames is not None:
            whisper_states = self.whisper_proj(whisper_frames)
            risk_scores = frame_risks

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            hidden_states = self.lm.model.embed_tokens(generated)

            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            if hasattr(self.lm.model, 'rotary_emb'):
                position_embeddings = self.lm.model.rotary_emb(hidden_states, position_ids)
            else:
                position_embeddings = None

            extra_kwargs = {}
            if position_embeddings is not None:
                extra_kwargs["position_embeddings"] = position_embeddings

            layers = self.lm.model.layers
            for i, layer in enumerate(layers):
                if str(i) in self.wrapped_layers:
                    wrapper = self.wrapped_layers[str(i)]
                    result = wrapper(
                        hidden_states,
                        whisper_states=whisper_states,
                        risk_scores=risk_scores,
                        whisper_mask=whisper_mask,
                        **extra_kwargs,
                    )
                else:
                    result = layer(hidden_states, **extra_kwargs)

                if isinstance(result, tuple):
                    hidden_states = result[0]
                else:
                    hidden_states = result

            if hasattr(self.lm.model, 'norm'):
                hidden_states = self.lm.model.norm(hidden_states)

            logits = self.lm.lm_head(hidden_states[:, -1:, :])

            if temperature != 1.0:
                logits = logits / temperature

            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.lm.config.eos_token_id:
                break

        return generated
