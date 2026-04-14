"""
AICAS 2026 - Participant Core Modification File

Optimized VLMModel for Qwen3-VL-2B-Instruct inference.
Optimizations:
- SDPA attention backend (auto-selects flash/mem_efficient/math per call)
- Fused decoder layers: Add+RMSNorm, QKV proj, gate_up proj, SwiGLU
- Step A decode path: custom greedy loop + StaticCache
"""
from collections import deque
from importlib import metadata as importlib_metadata
import platform
import sys
from types import MethodType
from typing import Dict
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import StaticCache

os.environ.setdefault("ENABLE_CUDA_NOSPLIT_GEMV", "1")
os.environ.setdefault("ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT", "1")
os.environ.setdefault("ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ", "1")
os.environ.setdefault("CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT", "persistent")

from optimized_layer import replace_decoder_layers, set_small_decode_triton_enabled, fused_vision_rope


_ENABLE_PREFILL_VISION_GRID_CACHE = os.environ.get("ENABLE_PREFILL_VISION_GRID_CACHE", "1") == "1"
_ENABLE_PREFILL_ROPE_FASTPATH = os.environ.get("ENABLE_PREFILL_ROPE_FASTPATH", "1") == "1"
_ENABLE_PREFILL_PATCH_EMBED_LINEAR = os.environ.get("ENABLE_PREFILL_PATCH_EMBED_LINEAR", "1") == "1"
_ENABLE_PREFILL_MM_SLICE_FASTPATH = os.environ.get("ENABLE_PREFILL_MM_SLICE_FASTPATH", "1") == "1"
_ENABLE_PREFILL_TEXT_FASTPATH = os.environ.get("ENABLE_PREFILL_TEXT_FASTPATH", "1") == "1"
_ENABLE_TRITON_PREFILL_FUSED_ADD_RMSNORM = os.environ.get("ENABLE_TRITON_PREFILL_FUSED_ADD_RMSNORM", "1") == "1"
_ENABLE_TRITON_PREFILL_FUSED_SILU_MUL = os.environ.get("ENABLE_TRITON_PREFILL_FUSED_SILU_MUL", "1") == "1"
_PREFILL_CACHE_MAX_ITEMS = max(1, int(os.environ.get("PREFILL_CACHE_MAX_ITEMS", "64")))


class CudaGraphExecutionError(RuntimeError):
    """Raised when CUDA Graph is enabled for a supported request but execution fails."""


class GreedyDecodeEngine:
    """Step A baseline: eager greedy decode on top of StaticCache."""

    SUPPORTED_KWARGS = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "use_cache",
        "eos_token_id",
        "pad_token_id",
    }

    def __init__(self, model, max_tokens_threshold: int):
        self._model = model
        self._max_tokens_threshold = max_tokens_threshold
        self._enable_cudagraph = os.environ.get("ENABLE_CUDAGRAPH", "1") == "1"
        self._graph_warmup_steps = int(os.environ.get("CUDAGRAPH_WARMUP_STEPS", "2"))
        self._async_eos_window = max(1, int(os.environ.get("CUDAGRAPH_ASYNC_EOS_WINDOW", "2")))
        self._graph_bucket_size = max(1, int(os.environ.get("CUDAGRAPH_BUCKET_SIZE", "64")))
        self._prebuild_max_cache_len = max(
            0,
            int(os.environ.get("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "2048")),
        )
        self._graph_runners: Dict[tuple[str, int], "DecodeCudaGraphRunner"] = {}
        self._prebuild_graph_runners()

    def is_supported(self, args, kwargs) -> bool:
        return self._get_support_failure_reason(args, kwargs) is None

    def _get_support_failure_reason(self, args, kwargs) -> str | None:
        if len(args) > 1:
            return f"len(args)={len(args)} > 1"

        unsupported_keys = set(kwargs) - self.SUPPORTED_KWARGS
        if unsupported_keys:
            return f"unsupported kwargs: {sorted(unsupported_keys)}"

        input_ids = args[0] if args else kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            return f"input_ids must be rank-2, got {None if input_ids is None else tuple(input_ids.shape)}"
        if input_ids.shape[0] != 1:
            return f"batch_size must be 1, got {input_ids.shape[0]}"

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.ndim != 2:
            return f"attention_mask must be rank-2, got shape={tuple(attention_mask.shape)}"

        if kwargs.get("do_sample", False):
            return "do_sample=True is unsupported"
        if not kwargs.get("use_cache", True):
            return "use_cache=False is unsupported"

        max_new_tokens = kwargs.get("max_new_tokens", self._max_tokens_threshold)
        if max_new_tokens is None or max_new_tokens < 1:
            return f"max_new_tokens must be >=1, got {max_new_tokens}"
        if max_new_tokens > self._max_tokens_threshold:
            return f"max_new_tokens={max_new_tokens} exceeds threshold={self._max_tokens_threshold}"

        return None

    def is_cudagraph_enabled(self) -> bool:
        return self._enable_cudagraph

    def should_use_cudagraph_for_request(self, args, kwargs) -> bool:
        if not self._enable_cudagraph:
            return False

        support_failure_reason = self._get_support_failure_reason(args, kwargs)
        if support_failure_reason is not None:
            return False

        input_ids = args[0] if args else kwargs["input_ids"]
        max_new_tokens = int(kwargs.get("max_new_tokens", self._max_tokens_threshold))
        return self._should_use_cudagraph(
            device=input_ids.device,
            max_new_tokens=max_new_tokens,
            current_length=input_ids.shape[1] + 1,
        )

    def generate(self, *args, **kwargs):
        input_ids = args[0] if args else kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        mm_token_type_ids = kwargs.get("mm_token_type_ids")
        max_new_tokens = int(kwargs.get("max_new_tokens", self._max_tokens_threshold))

        batch_size, prompt_length = input_ids.shape
        device = input_ids.device
        max_cache_len = prompt_length + max_new_tokens
        require_cudagraph = self.is_cudagraph_required_for_request(args, kwargs)

        if attention_mask is None:
            prompt_attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.bool, device=device)
        else:
            prompt_attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        graph_runner = None
        graph_bucket_len = None
        full_attention_mask = None
        static_cache = None

        if self._should_use_cudagraph(
            device=device,
            max_new_tokens=max_new_tokens,
            current_length=prompt_length + 1,
        ):
            graph_bucket_len = self._get_graph_bucket_len(max_cache_len)
            try:
                graph_runner = self._get_or_create_graph_runner(
                    device=device,
                    bucket_max_cache_len=graph_bucket_len,
                )
                graph_runner.prepare_for_request(prompt_attention_mask)
                full_attention_mask = graph_runner.attention_mask
                static_cache = graph_runner.static_cache
            except Exception as exc:
                if graph_bucket_len is not None:
                    self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                if require_cudagraph:
                    self._raise_cudagraph_failure(
                        stage="runner_setup",
                        exc=exc,
                        device=device,
                        prompt_length=prompt_length,
                        current_length=prompt_length + 1,
                        max_new_tokens=max_new_tokens,
                        graph_bucket_len=graph_bucket_len,
                    )
                print(f"[GreedyDecodeEngine] CUDA Graph runner setup failed, fallback to eager decode: {exc}")
                graph_runner = None
                graph_bucket_len = None

        if graph_runner is None:
            full_attention_mask = torch.zeros((batch_size, max_cache_len), dtype=torch.bool, device=device)
            full_attention_mask[:, :prompt_length] = prompt_attention_mask
            static_cache = StaticCache(
                config=self._model.config,
                max_cache_len=max_cache_len,
            )

        output_ids = torch.empty(
            (batch_size, max_cache_len),
            dtype=input_ids.dtype,
            device=device,
        )
        output_ids[:, :prompt_length] = input_ids

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                past_key_values=static_cache,
                use_cache=True,
                logits_to_keep=1,
            )

            rope_deltas = outputs.rope_deltas
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            if graph_runner is not None:
                try:
                    graph_runner.capture_if_needed(
                        rope_deltas=rope_deltas,
                        start_position=prompt_length,
                        start_input_ids=next_tokens,
                    )
                except Exception as exc:
                    if graph_bucket_len is not None:
                        self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                    if require_cudagraph:
                        self._raise_cudagraph_failure(
                            stage="capture",
                            exc=exc,
                            device=device,
                            prompt_length=prompt_length,
                            current_length=prompt_length + 1,
                            max_new_tokens=max_new_tokens,
                            graph_bucket_len=graph_bucket_len,
                        )
                    print(f"[GreedyDecodeEngine] CUDA Graph capture failed, fallback to eager decode: {exc}")
                    graph_runner = None

            output_ids[:, prompt_length:prompt_length + 1] = next_tokens

            current_length = prompt_length + 1
            current_position = prompt_length
            eos_token_id = self._resolve_eos_token_id(kwargs)
            eos_token_values = self._resolve_eos_token_values(kwargs)

            if graph_runner is not None and eos_token_values and self._async_eos_window > 1:
                return self._generate_with_async_eos_poll(
                    next_tokens=next_tokens,
                    output_ids=output_ids,
                    current_length=current_length,
                    current_position=current_position,
                    max_new_tokens=max_new_tokens,
                    graph_runner=graph_runner,
                    eos_token_values=eos_token_values,
                    device=device,
                    prompt_length=prompt_length,
                    graph_bucket_len=graph_bucket_len,
                    require_cudagraph=require_cudagraph,
                )

            for _ in range(1, max_new_tokens):
                if eos_token_id is not None and bool((next_tokens.squeeze() == eos_token_id).any()):
                    break

                if graph_runner is not None:
                    try:
                        next_tokens = graph_runner.replay(
                            input_ids=next_tokens,
                            current_position=current_position,
                        )
                    except Exception as exc:
                        if graph_bucket_len is not None:
                            self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                        if require_cudagraph:
                            self._raise_cudagraph_failure(
                                stage="replay",
                                exc=exc,
                                device=device,
                                prompt_length=prompt_length,
                                current_length=current_length,
                                max_new_tokens=max_new_tokens,
                                graph_bucket_len=graph_bucket_len,
                            )
                        raise
                else:
                    next_tokens = self._decode_one_token_eager(
                        next_tokens=next_tokens,
                        full_attention_mask=full_attention_mask,
                        rope_deltas=rope_deltas,
                        static_cache=static_cache,
                        current_position=current_position,
                        device=device,
                    )
                output_ids[:, current_length:current_length + 1] = next_tokens
                current_length += 1
                current_position += 1

        return output_ids[:, :current_length]

    def _generate_with_async_eos_poll(
        self,
        next_tokens: torch.Tensor,
        output_ids: torch.Tensor,
        current_length: int,
        current_position: int,
        max_new_tokens: int,
        graph_runner,
        eos_token_values,
        device: torch.device,
        prompt_length: int,
        graph_bucket_len: int | None,
        require_cudagraph: bool,
    ):
        pending = deque()
        try:
            pending.append(graph_runner.stage_host_token(next_tokens, output_length=current_length))
        except Exception as exc:
            if graph_bucket_len is not None:
                self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
            if require_cudagraph:
                self._raise_cudagraph_failure(
                    stage="async_stage_host_token",
                    exc=exc,
                    device=device,
                    prompt_length=prompt_length,
                    current_length=current_length,
                    max_new_tokens=max_new_tokens,
                    graph_bucket_len=graph_bucket_len,
                )
            raise

        for _ in range(1, max_new_tokens):
            try:
                eos_length = graph_runner.poll_staged_tokens(
                    pending=pending,
                    eos_token_values=eos_token_values,
                    block=False,
                )
            except Exception as exc:
                if graph_bucket_len is not None:
                    self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                if require_cudagraph:
                    self._raise_cudagraph_failure(
                        stage="async_poll",
                        exc=exc,
                        device=device,
                        prompt_length=prompt_length,
                        current_length=current_length,
                        max_new_tokens=max_new_tokens,
                        graph_bucket_len=graph_bucket_len,
                    )
                raise
            if eos_length is not None:
                current_length = eos_length
                break

            if len(pending) >= graph_runner.async_eos_window:
                try:
                    eos_length = graph_runner.poll_staged_tokens(
                        pending=pending,
                        eos_token_values=eos_token_values,
                        block=True,
                    )
                except Exception as exc:
                    if graph_bucket_len is not None:
                        self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                    if require_cudagraph:
                        self._raise_cudagraph_failure(
                            stage="async_poll_blocking",
                            exc=exc,
                            device=device,
                            prompt_length=prompt_length,
                            current_length=current_length,
                            max_new_tokens=max_new_tokens,
                            graph_bucket_len=graph_bucket_len,
                        )
                    raise
                if eos_length is not None:
                    current_length = eos_length
                    break

            try:
                next_tokens = graph_runner.replay(
                    input_ids=next_tokens,
                    current_position=current_position,
                )
            except Exception as exc:
                if graph_bucket_len is not None:
                    self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                if require_cudagraph:
                    self._raise_cudagraph_failure(
                        stage="async_replay",
                        exc=exc,
                        device=device,
                        prompt_length=prompt_length,
                        current_length=current_length,
                        max_new_tokens=max_new_tokens,
                        graph_bucket_len=graph_bucket_len,
                    )
                raise
            output_ids[:, current_length:current_length + 1] = next_tokens
            current_length += 1
            current_position += 1
            try:
                pending.append(graph_runner.stage_host_token(next_tokens, output_length=current_length))
            except Exception as exc:
                if graph_bucket_len is not None:
                    self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                if require_cudagraph:
                    self._raise_cudagraph_failure(
                        stage="async_stage_host_token",
                        exc=exc,
                        device=device,
                        prompt_length=prompt_length,
                        current_length=current_length,
                        max_new_tokens=max_new_tokens,
                        graph_bucket_len=graph_bucket_len,
                    )
                raise

        if pending:
            try:
                eos_length = graph_runner.poll_staged_tokens(
                    pending=pending,
                    eos_token_values=eos_token_values,
                    block=True,
                    drain_all=True,
                )
            except Exception as exc:
                if graph_bucket_len is not None:
                    self._graph_runners.pop(self._graph_runner_key(device, graph_bucket_len), None)
                if require_cudagraph:
                    self._raise_cudagraph_failure(
                        stage="async_drain",
                        exc=exc,
                        device=device,
                        prompt_length=prompt_length,
                        current_length=current_length,
                        max_new_tokens=max_new_tokens,
                        graph_bucket_len=graph_bucket_len,
                    )
                raise
            if eos_length is not None:
                current_length = eos_length

        return output_ids[:, :current_length]

    def is_cudagraph_required_for_request(self, args, kwargs) -> bool:
        return self.should_use_cudagraph_for_request(args, kwargs)

    def _resolve_eos_token_id(self, kwargs):
        eos_token_id = kwargs.get("eos_token_id")
        if eos_token_id is None:
            gen_cfg = getattr(self._model, "generation_config", None)
            if gen_cfg is not None:
                eos_token_id = getattr(gen_cfg, "eos_token_id", None)
        if eos_token_id is None:
            text_config = self._model.config.get_text_config(decoder=True)
            eos_token_id = getattr(text_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            if not eos_token_id:
                return None
            return torch.tensor(eos_token_id, dtype=torch.long, device=next(self._model.parameters()).device)
        if eos_token_id is not None:
            return torch.tensor([eos_token_id], dtype=torch.long, device=next(self._model.parameters()).device)
        return None

    def _resolve_eos_token_values(self, kwargs):
        eos_token_id = kwargs.get("eos_token_id")
        if eos_token_id is None:
            gen_cfg = getattr(self._model, "generation_config", None)
            if gen_cfg is not None:
                eos_token_id = getattr(gen_cfg, "eos_token_id", None)
        if eos_token_id is None:
            text_config = self._model.config.get_text_config(decoder=True)
            eos_token_id = getattr(text_config, "eos_token_id", None)

        if eos_token_id is None:
            return None
        if isinstance(eos_token_id, torch.Tensor):
            eos_token_id = eos_token_id.detach().cpu().view(-1).tolist()
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_token_id = list(eos_token_id)
        else:
            eos_token_id = [eos_token_id]

        values = tuple(int(token_id) for token_id in eos_token_id)
        return values if values else None

    def _build_decode_position_ids(self, current_position: int, rope_deltas, device: torch.device):
        text_position_ids = torch.full((1, 1), current_position, dtype=torch.long, device=device)
        if rope_deltas is not None:
            text_position_ids = text_position_ids + rope_deltas.to(device=device, dtype=torch.long).view(1, 1)
        return text_position_ids.view(1, 1, 1).expand(3, 1, 1)

    def _decode_one_token_eager(
        self,
        next_tokens: torch.Tensor,
        full_attention_mask: torch.Tensor,
        rope_deltas,
        static_cache: StaticCache,
        current_position: int,
        device: torch.device,
    ):
        full_attention_mask[:, current_position] = True
        position_ids = self._build_decode_position_ids(
            current_position=current_position,
            rope_deltas=rope_deltas,
            device=device,
        )
        cache_position = torch.tensor([current_position], dtype=torch.long, device=device)

        outputs = self._model(
            input_ids=next_tokens,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            past_key_values=static_cache,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        return torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    def _extract_request_input_ids(self, args, kwargs):
        input_ids = args[0] if args else kwargs.get("input_ids")
        if input_ids is None or getattr(input_ids, "ndim", None) != 2:
            return None
        return input_ids

    def _extract_request_device(self, args, kwargs):
        input_ids = self._extract_request_input_ids(args, kwargs)
        if input_ids is not None:
            return input_ids.device
        try:
            return next(self._model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _build_request_context(self, args, kwargs):
        input_ids = args[0] if args else kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        return "\n".join(
            [
                f"args_len={len(args)}",
                f"kwargs_keys={sorted(kwargs.keys())}",
                f"input_ids_shape={None if input_ids is None else tuple(input_ids.shape)}",
                f"input_ids_device={None if input_ids is None else input_ids.device}",
                f"attention_mask_shape={None if attention_mask is None else tuple(attention_mask.shape)}",
                f"max_new_tokens={kwargs.get('max_new_tokens', self._max_tokens_threshold)}",
                f"use_cache={kwargs.get('use_cache', True)}",
                f"do_sample={kwargs.get('do_sample', False)}",
            ]
        )

    def _get_cudagraph_unavailable_reason(
        self,
        device: torch.device,
        max_new_tokens: int,
        current_length: int,
    ) -> str | None:
        if not self._enable_cudagraph:
            return "ENABLE_CUDAGRAPH=0"
        if not torch.cuda.is_available():
            return "torch.cuda.is_available()=False"
        if device.type != "cuda":
            return f"device.type={device.type}, expected cuda"
        if max_new_tokens <= 1:
            return f"max_new_tokens={max_new_tokens} <= 1"
        if current_length < 1:
            return f"current_length={current_length} < 1"
        return None

    def _should_use_cudagraph(self, device: torch.device, max_new_tokens: int, current_length: int):
        return self._get_cudagraph_unavailable_reason(
            device=device,
            max_new_tokens=max_new_tokens,
            current_length=current_length,
        ) is None

    def _get_graph_bucket_len(self, max_cache_len: int):
        bucket_size = self._graph_bucket_size
        return ((max_cache_len + bucket_size - 1) // bucket_size) * bucket_size

    def _graph_runner_key(self, device: torch.device, bucket_max_cache_len: int):
        return str(device), bucket_max_cache_len

    def _get_or_create_graph_runner(self, device: torch.device, bucket_max_cache_len: int):
        key = self._graph_runner_key(device, bucket_max_cache_len)
        runner = self._graph_runners.get(key)
        if runner is None:
            runner = DecodeCudaGraphRunner(
                model=self._model,
                device=device,
                max_cache_len=bucket_max_cache_len,
                warmup_steps=self._graph_warmup_steps,
                async_eos_window=self._async_eos_window,
            )
            self._graph_runners[key] = runner
        return runner

    def _raise_cudagraph_failure(
        self,
        stage: str,
        exc: Exception,
        device: torch.device,
        prompt_length: int,
        current_length: int,
        max_new_tokens: int,
        graph_bucket_len: int | None,
    ):
        context = [
            "CUDA Graph execution failed while ENABLE_CUDAGRAPH=1 and request passed _should_use_cudagraph().",
            f"stage={stage}",
            f"device={device}",
            f"prompt_length={prompt_length}",
            f"current_length={current_length}",
            f"max_new_tokens={max_new_tokens}",
            f"graph_bucket_len={graph_bucket_len}",
            f"original_exception={type(exc).__name__}: {exc}",
        ]
        raise CudaGraphExecutionError(
            "\n".join(context + ["", self._build_runtime_environment_report(device=device)])
        ) from exc

    def _build_runtime_environment_report(self, device: torch.device):
        env_names = (
            "ENABLE_CUDAGRAPH",
            "CUDAGRAPH_WARMUP_STEPS",
            "CUDAGRAPH_ASYNC_EOS_WINDOW",
            "CUDAGRAPH_BUCKET_SIZE",
            "CUDAGRAPH_PREBUILD_MAX_CACHE_LEN",
            "ENABLE_DECODE_NVTX",
            "ENABLE_TRITON_SMALL_RMSNORM",
            "ENABLE_TRITON_SMALL_ADD_RMSNORM",
            "ENABLE_TRITON_SMALL_QK_NORM_ROPE",
            "ENABLE_TRITON_SMALL_SILU_MUL",
            "ENABLE_CUSTOM_DECODE",
            "ENABLE_VIZTRACE",
            "VIZTRACE_OUTPUT_DIR",
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
        )
        package_names = (
            "torch",
            "transformers",
            "tokenizers",
            "safetensors",
            "accelerate",
            "flash_attn",
            "triton",
            "numpy",
            "Pillow",
        )

        lines = ["Runtime environment check:"]
        lines.append(f"python={platform.python_version()} ({sys.executable})")
        lines.append(f"platform={platform.platform()}")
        lines.append(f"torch.__version__={torch.__version__}")
        lines.append(f"torch.version.cuda={torch.version.cuda}")
        lines.append(f"torch.backends.cudnn.version={torch.backends.cudnn.version()}")
        lines.append(f"torch.cuda.is_available={torch.cuda.is_available()}")
        lines.append(f"torch.cuda.device_count={torch.cuda.device_count()}")
        lines.append(f"requested_device={device}")
        lines.append(f"model_dtype={getattr(self._model, 'dtype', 'unknown')}")
        lines.append(f"model_type={getattr(self._model.config, 'model_type', 'unknown')}")

        if torch.cuda.is_available():
            try:
                device_index = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                lines.append(
                    "cuda_device="
                    f"{props.name}, capability={props.major}.{props.minor}, total_memory_gb={props.total_memory / (1024 ** 3):.2f}"
                )
            except Exception as cuda_exc:
                lines.append(f"cuda_device=<unavailable: {type(cuda_exc).__name__}: {cuda_exc}>")

        for package_name in package_names:
            try:
                version = importlib_metadata.version(package_name)
            except importlib_metadata.PackageNotFoundError:
                version = "not_installed"
            except Exception as package_exc:
                version = f"error:{type(package_exc).__name__}:{package_exc}"
            lines.append(f"{package_name}={version}")

        for env_name in env_names:
            lines.append(f"{env_name}={os.environ.get(env_name, '<unset>')}")

        return "\n".join(lines)

    def _prebuild_graph_runners(self):
        if not self._enable_cudagraph:
            return
        if self._prebuild_max_cache_len < self._graph_bucket_size:
            return
        if not torch.cuda.is_available():
            return

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        buckets = range(
            self._graph_bucket_size,
            self._prebuild_max_cache_len + 1,
            self._graph_bucket_size,
        )
        print(
            "[GreedyDecodeEngine] Prebuilding CUDA graph buckets "
            f"{self._graph_bucket_size}..{self._prebuild_max_cache_len} "
            f"(step={self._graph_bucket_size})"
        )
        with torch.no_grad():
            for bucket_max_cache_len in buckets:
                try:
                    runner = self._get_or_create_graph_runner(
                        device=device,
                        bucket_max_cache_len=bucket_max_cache_len,
                    )
                    runner.prebuild_graph()
                except Exception as exc:
                    self._graph_runners.pop(self._graph_runner_key(device, bucket_max_cache_len), None)
                    print(
                        "[GreedyDecodeEngine] Prebuild failed for bucket "
                        f"{bucket_max_cache_len}, fallback to lazy capture: {exc}"
                    )


class DecodeCudaGraphRunner:
    """Capture and replay a single-token decode step with fixed buffers."""

    def __init__(
        self,
        model,
        device: torch.device,
        max_cache_len: int,
        warmup_steps: int = 2,
        async_eos_window: int = 2,
    ):
        self._model = model
        self._device = device
        self._max_cache_len = max_cache_len
        self._warmup_steps = max(1, warmup_steps)
        self._async_eos_window = max(1, async_eos_window)
        self._static_cache = StaticCache(
            config=model.config,
            max_cache_len=max_cache_len,
        )
        self._attention_mask_buf = torch.zeros((1, max_cache_len), dtype=torch.bool, device=self._device)
        self._cache_position_buf = torch.empty((1,), dtype=torch.long, device=self._device)
        self._cache_position_table = torch.arange(max_cache_len, dtype=torch.long, device=self._device)
        self._rope_delta_buf = torch.zeros((1, 1), dtype=torch.long, device=self._device)
        self._input_ids_buf = torch.empty((1, 1), dtype=torch.long, device=self._device)
        self._position_ids_buf = torch.empty((3, 1, 1), dtype=torch.long, device=self._device)
        vocab_size = model.config.get_text_config(decoder=True).vocab_size
        self._logits_buf = torch.empty((1, vocab_size), dtype=model.dtype, device=self._device)
        self._next_token_buf = torch.empty((1, 1), dtype=torch.long, device=self._device)
        self._token_stage_bufs = [
            torch.empty((1, 1), dtype=torch.long, device=self._device)
            for _ in range(self._async_eos_window)
        ]
        self._host_token_bufs = [
            torch.empty((1, 1), dtype=torch.long, device="cpu", pin_memory=True)
            for _ in range(self._async_eos_window)
        ]
        self._host_token_events = [torch.cuda.Event() for _ in range(self._async_eos_window)]
        self._next_stage_slot = 0
        self._graph = None

    def replay(self, input_ids: torch.Tensor, current_position: int):
        self._write_inputs(input_ids, current_position)
        self._graph.replay()
        return self._next_token_buf.clone()

    @property
    def async_eos_window(self):
        return self._async_eos_window

    @property
    def static_cache(self):
        return self._static_cache

    @property
    def attention_mask(self):
        return self._attention_mask_buf

    def prepare_for_request(self, prompt_attention_mask: torch.Tensor):
        prompt_length = prompt_attention_mask.shape[1]
        if prompt_length > self._max_cache_len:
            raise ValueError(
                f"prompt_length={prompt_length} exceeds graph bucket max_cache_len={self._max_cache_len}"
            )

        self._static_cache.reset()
        self._attention_mask_buf.zero_()
        self._attention_mask_buf[:, :prompt_length].copy_(prompt_attention_mask)
        self._rope_delta_buf.zero_()
        self._next_stage_slot = 0

    def capture_if_needed(self, rope_deltas, start_position: int, start_input_ids: torch.Tensor):
        self._set_rope_deltas(rope_deltas)
        if self._graph is not None:
            return

        previous_small_decode_triton = set_small_decode_triton_enabled(self._model, True)
        try:
            self._graph = torch.cuda.CUDAGraph()
            cache_snapshot = self._snapshot_cache()
            self._write_inputs(start_input_ids, start_position)

            side_stream = torch.cuda.Stream(device=self._device)
            with torch.cuda.stream(side_stream):
                for _ in range(self._warmup_steps):
                    self._run_step()
                    self._restore_cache(cache_snapshot)
                    self._write_inputs(start_input_ids, start_position)
            torch.cuda.current_stream(device=self._device).wait_stream(side_stream)

            self._restore_cache(cache_snapshot)
            self._write_inputs(start_input_ids, start_position)
            torch.cuda.synchronize(self._device)
            with torch.cuda.graph(self._graph):
                self._run_step()
            torch.cuda.synchronize(self._device)
            self._restore_cache(cache_snapshot)
            self._write_inputs(start_input_ids, start_position)
        finally:
            set_small_decode_triton_enabled(self._model, previous_small_decode_triton)

    def prebuild_graph(self):
        if self._graph is not None:
            return

        previous_small_decode_triton = set_small_decode_triton_enabled(self._model, True)
        try:
            text_config = self._model.config.get_text_config(decoder=True)
            self._static_cache.early_initialization(
                batch_size=1,
                num_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                dtype=self._model.dtype,
                device=self._device,
            )
            self._static_cache.reset()
            self._attention_mask_buf.zero_()
            self._rope_delta_buf.zero_()
            self._write_inputs(self._input_ids_buf.zero_(), current_position=0)

            self._graph = torch.cuda.CUDAGraph()
            side_stream = torch.cuda.Stream(device=self._device)
            with torch.cuda.stream(side_stream):
                for _ in range(self._warmup_steps):
                    self._run_step()
                    self._static_cache.reset()
                    self._attention_mask_buf.zero_()
                    self._write_inputs(self._input_ids_buf.zero_(), current_position=0)
            torch.cuda.current_stream(device=self._device).wait_stream(side_stream)

            self._static_cache.reset()
            self._attention_mask_buf.zero_()
            self._write_inputs(self._input_ids_buf.zero_(), current_position=0)
            torch.cuda.synchronize(self._device)
            with torch.cuda.graph(self._graph):
                self._run_step()
            torch.cuda.synchronize(self._device)
            self._static_cache.reset()
            self._attention_mask_buf.zero_()
            self._write_inputs(self._input_ids_buf.zero_(), current_position=0)
        finally:
            set_small_decode_triton_enabled(self._model, previous_small_decode_triton)

    def stage_host_token(self, device_token: torch.Tensor, output_length: int):
        slot = self._next_stage_slot
        self._next_stage_slot = (self._next_stage_slot + 1) % self._async_eos_window
        staged_token = self._token_stage_bufs[slot]
        host_token = self._host_token_bufs[slot]
        event = self._host_token_events[slot]

        staged_token.copy_(device_token)
        host_token.copy_(staged_token, non_blocking=True)
        event.record(torch.cuda.current_stream(device=self._device))
        return {"slot": slot, "output_length": output_length}

    def poll_staged_tokens(self, pending, eos_token_values, block: bool, drain_all: bool = False):
        eos_set = set(eos_token_values)
        eos_length = None

        while pending:
            record = pending[0]
            event = self._host_token_events[record["slot"]]
            if block:
                event.synchronize()
            elif not event.query():
                break

            token = int(self._host_token_bufs[record["slot"]].item())
            pending.popleft()
            if token in eos_set and eos_length is None:
                eos_length = record["output_length"]
                if not drain_all:
                    return eos_length
            elif not drain_all:
                break

        return eos_length

    def _run_step(self):
        outputs = self._model(
            input_ids=self._input_ids_buf,
            attention_mask=self._attention_mask_buf,
            position_ids=self._position_ids_buf,
            past_key_values=self._static_cache,
            use_cache=True,
            cache_position=self._cache_position_buf,
            logits_to_keep=1,
        )
        self._logits_buf.copy_(outputs.logits[:, -1, :])
        self._next_token_buf.copy_(torch.argmax(self._logits_buf, dim=-1, keepdim=True))

    def _write_inputs(self, input_ids: torch.Tensor, current_position: int):
        self._input_ids_buf.copy_(input_ids)
        self._attention_mask_buf[:, current_position] = True
        self._cache_position_buf.copy_(self._cache_position_table.narrow(0, current_position, 1))
        position_id = self._cache_position_table.narrow(0, current_position, 1).view(1, 1, 1)
        self._position_ids_buf.copy_(position_id.expand(3, 1, 1))
        self._position_ids_buf.add_(self._rope_delta_buf.view(1, 1, 1))

    def _set_rope_deltas(self, rope_deltas):
        if rope_deltas is None:
            self._rope_delta_buf.zero_()
            return
        self._rope_delta_buf.copy_(rope_deltas.to(device=self._device, dtype=torch.long).view(1, 1))

    def _snapshot_cache(self):
        snapshot = []
        for layer in self._static_cache.layers:
            layer_snapshot = {
                "keys": None if layer.keys is None else layer.keys.clone(),
                "values": None if layer.values is None else layer.values.clone(),
            }
            if hasattr(layer, "cumulative_length"):
                cumulative_length = layer.cumulative_length
                if torch.is_tensor(cumulative_length):
                    layer_snapshot["cumulative_length"] = cumulative_length.clone()
                else:
                    layer_snapshot["cumulative_length"] = cumulative_length
            snapshot.append(layer_snapshot)
        return snapshot

    def _restore_cache(self, snapshot):
        for layer, layer_snapshot in zip(self._static_cache.layers, snapshot):
            keys = layer_snapshot["keys"]
            values = layer_snapshot["values"]
            if keys is not None:
                layer.keys.copy_(keys)
            if values is not None:
                layer.values.copy_(values)

            if "cumulative_length" not in layer_snapshot or not hasattr(layer, "cumulative_length"):
                continue

            cumulative_length = layer_snapshot["cumulative_length"]
            if torch.is_tensor(layer.cumulative_length):
                if torch.is_tensor(cumulative_length):
                    layer.cumulative_length.copy_(cumulative_length)
                else:
                    layer.cumulative_length.fill_(cumulative_length)
            elif torch.is_tensor(cumulative_length):
                layer.cumulative_length = int(cumulative_length.item())
            else:
                layer.cumulative_length = cumulative_length


class ModelWrapper:
    """Transparent wrapper around the HF model with optional tracing."""

    WARMUP_CALLS = 10
    MAX_TOKENS_THRESHOLD = 128

    def __init__(self, model):
        self._model = model
        self._enable_trace = os.environ.get("ENABLE_VIZTRACE", "0") == "1"
        self._enable_custom_decode = os.environ.get("ENABLE_CUSTOM_DECODE", "1") == "1"
        self._warmup_calls = int(os.environ.get("VIZTRACE_WARMUP_CALLS", str(self.WARMUP_CALLS)))
        self._call_count = 0
        self._trace_count = 0
        self._custom_decode = GreedyDecodeEngine(model, self.MAX_TOKENS_THRESHOLD)
        if self._enable_trace:
            from viztracer import VizTracer  # noqa: F401
            self._trace_dir = os.environ.get(
                "VIZTRACE_OUTPUT_DIR",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces"),
            )
            os.makedirs(self._trace_dir, exist_ok=True)
            print(f"[ModelWrapper] Trace enabled, output: {self._trace_dir}")

    def _should_trace(self, kwargs):
        if self._call_count <= self._warmup_calls:
            return False
        if kwargs.get("max_new_tokens", 0) > self.MAX_TOKENS_THRESHOLD:
            return False
        return True

    def generate(self, *args, **kwargs):
        self._call_count += 1
        use_custom_decode = False
        generate_fn = lambda: self._model.generate(*args, **kwargs)
        cudagraph_required = self._custom_decode.should_use_cudagraph_for_request(args, kwargs)

        if cudagraph_required and not self._enable_custom_decode:
            raise CudaGraphExecutionError(
                "\n".join(
                    [
                        "ENABLE_CUDAGRAPH=1 and this request should use CUDA Graph, but ENABLE_CUSTOM_DECODE=0.",
                        "",
                        self._custom_decode._build_request_context(args, kwargs),
                        "",
                        self._custom_decode._build_runtime_environment_report(
                            device=self._custom_decode._extract_request_device(args, kwargs)
                        ),
                    ]
                )
            )

        if self._enable_custom_decode and self._custom_decode.is_supported(args, kwargs):
            use_custom_decode = True
            generate_fn = lambda: self._custom_decode.generate(*args, **kwargs)

        if not self._enable_trace or not self._should_trace(kwargs):
            try:
                return generate_fn()
            except Exception:
                raise

        self._trace_count += 1
        from viztracer import VizTracer
        trace_file = os.path.join(self._trace_dir, f"generate_{self._trace_count}.json")
        tracer = VizTracer(log_torch=True, output_file=trace_file, max_stack_depth=15, verbose=0)
        tracer.start()
        try:
            try:
                result = generate_fn()
            except Exception:
                raise
        finally:
            tracer.stop()
            tracer.save()
            print(f"[ModelWrapper] Trace saved: {trace_file}")
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)


class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self._device = device
        self.model_path = model_path
        self._optimizations_applied = []

        import time as _time

        # CUDA backend config
        self._configure_cuda()

        # Load processor
        t0 = _time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(model_path)
        print(f"[VLMModel] Processor loaded in {_time.perf_counter()-t0:.2f}s")

        # Load model with SDPA (auto-selects flash/mem_efficient/math per call)
        attn_impl = "sdpa"
        print(f"[VLMModel] Loading model with attn_implementation={attn_impl}...")
        t0 = _time.perf_counter()
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation=attn_impl,
        )
        self._model.eval()
        print(f"[VLMModel] Model loaded in {_time.perf_counter()-t0:.2f}s")
        self._optimizations_applied.append(f"attn:{attn_impl}")

        # KV cache config
        self._optimize_kv_cache()

        # Replace decoder layers with fused versions
        self._replace_decoder_layers()

        # Initialize HBM partition-aware GEMV if enabled
        self._init_partition_gemv()

        # Prefill-specific path optimizations
        self._optimize_prefill_path()

        # Wrap model
        self._wrapped_model = ModelWrapper(self._model)
        if os.environ.get("ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_SOFTSYNC", "0") == "1":
            self._optimizations_applied.append("decode:o_proj_post_attn_mlp_softsync")
        elif os.environ.get("ENABLE_CUDA_FUSED_O_PROJ_POST_ATTN_MLP_PERSISTENT", "0") == "1":
            self._optimizations_applied.append("decode:o_proj_post_attn_mlp_persistent")
        if os.environ.get("ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT", "0") == "1":
            self._optimizations_applied.append("decode:post_attn_mlp_persistent")
        if os.environ.get("ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ", "0") == "1":
            variant = os.environ.get("CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT", "persistent").strip().lower()
            self._optimizations_applied.append(f"decode:mlp_fused_{variant}")

        print(f"[VLMModel] Optimizations: {', '.join(self._optimizations_applied)}")

    def _configure_cuda(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.set_float32_matmul_precision('high')
        self._optimizations_applied.append("cuda_config")

    def _optimize_kv_cache(self):
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id'):
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._model.config.eos_token_id
        self._optimizations_applied.append("kv_cache")

    def _replace_decoder_layers(self):
        """Replace HF decoder layers with fused OptimizedDecoderLayer."""
        import time as _time
        t0 = _time.perf_counter()
        try:
            count = replace_decoder_layers(self._model)
            elapsed = _time.perf_counter() - t0
            print(f"[VLMModel] Replaced {count} decoder layers with fused versions in {elapsed:.2f}s")
            self._optimizations_applied.append(f"fused_layers:{count}")
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to replace decoder layers: {e}")
            import traceback
            traceback.print_exc()

    def _init_partition_gemv(self):
        """Initialize HBM partition-aware GEMV for k=2048 weights."""
        if os.environ.get("ENABLE_PARTITION_GEMV", "0") != "1":
            return
        import time as _time
        t0 = _time.perf_counter()
        try:
            import optimized_layer
            from hbm_latency_partition import (
                create_partition_buffer, pack_interleaved_weight,
                compute_interleaved_size,
            )

            model_core = getattr(self._model, "model", None)
            lang_model = getattr(model_core, "language_model", model_core) if model_core is not None else None
            if lang_model is None or not hasattr(lang_model, "layers"):
                print("[VLMModel] Partition GEMV: no decoder layers found, skipping")
                return

            weight_shapes = []
            weight_refs = []
            for layer in lang_model.layers:
                if not hasattr(layer, "qkv_weight"):
                    continue
                for name, attr_name in [("qkv", "qkv_weight"), ("o_proj", "o_proj_weight"), ("gate_up", "gate_up_weight")]:
                    w = getattr(layer, attr_name, None)
                    if w is not None and w.shape[1] == 2048:
                        weight_shapes.append(tuple(w.shape))
                        weight_refs.append((layer.layer_idx, name, w))

            if not weight_shapes:
                print("[VLMModel] Partition GEMV: no k=2048 weights found, skipping")
                return

            print(f"[VLMModel] Partition GEMV: packing {len(weight_shapes)} weights...")
            pd = create_partition_buffer(weight_shapes, device=str(self._device))

            partition_weights = {}
            for i, (layer_idx, name, w) in enumerate(weight_refs):
                n_rows, k = weight_shapes[i]
                buf_size = compute_interleaved_size(n_rows, k)
                buf_offset_halves = pd["offsets"][i] // 2
                buf_slice = pd["buffer"][buf_offset_halves: buf_offset_halves + buf_size // 2]
                pack_interleaved_weight(w, buf_slice, pd["chunk_classes_list"][i])
                key = f"layer{layer_idx}.{name}"
                partition_weights[key] = {
                    "buf": buf_slice,
                    "chunk_classes": pd["chunk_classes_list"][i].to(self._device),
                    "n_rows": n_rows,
                }

            optimized_layer._PARTITION_DATA = {
                "sm_class_lo": pd["sm_class_lo"],
                "sm_class_hi": pd["sm_class_hi"],
                "weights": partition_weights,
                "_buffer": pd["buffer"],
            }

            # Register an atexit handler (runs LIFO, so first-registered runs last)
            # that hard-exits to avoid CUDA segfault during Python module teardown
            # when the large partition buffer storage gets freed after CUDA context
            # is already destroyed.
            import atexit
            def _hard_exit_to_avoid_cuda_segfault():
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            atexit.register(_hard_exit_to_avoid_cuda_segfault)

            elapsed = _time.perf_counter() - t0
            total_mb = sum(compute_interleaved_size(*s) for s in weight_shapes) / (1024 * 1024)
            print(f"[VLMModel] Partition GEMV: {len(partition_weights)} weights packed "
                  f"({total_mb:.0f} MB buffer) in {elapsed:.2f}s")
            self._optimizations_applied.append(f"partition_gemv:{len(partition_weights)}")
        except Exception as e:
            print(f"[VLMModel] Warning: Failed to init partition GEMV: {e}")
            import traceback
            traceback.print_exc()

    def _optimize_prefill_path(self):
        """Install low-risk prefill-specific caches and benchmark-pattern fast paths."""
        model_core = getattr(self._model, "model", None)
        if model_core is None:
            return
        visual = getattr(model_core, "visual", None)
        visual_dtype = getattr(visual, "dtype", None)
        spatial_merge_size = int(model_core.config.vision_config.spatial_merge_size)

        def _cache_put(cache, key, value):
            if key in cache:
                cache.pop(key)
            cache[key] = value
            if len(cache) > _PREFILL_CACHE_MAX_ITEMS:
                cache.pop(next(iter(cache)))
            return value

        prefill_grid_key_by_ptr = {}
        prefill_segment_info_by_ptr = {}
        def _tensor_ptr(tensor):
            if torch.is_tensor(tensor):
                return int(tensor.data_ptr())
            return None

        def _segment_ptr_key(
            input_ids: torch.Tensor | None,
            mm_token_type_ids: torch.Tensor | None,
            image_grid_thw: torch.Tensor | None,
            video_grid_thw: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
        ):
            if (
                not torch.is_tensor(input_ids)
                or not torch.is_tensor(mm_token_type_ids)
                or not torch.is_tensor(image_grid_thw)
                or video_grid_thw is not None
            ):
                return None
            return (
                int(input_ids.data_ptr()),
                int(mm_token_type_ids.data_ptr()),
                int(image_grid_thw.data_ptr()),
            )

        def _grid_key_from_rows(rows):
            if rows is None:
                return ()
            if rows and not isinstance(rows[0], (list, tuple)):
                rows = [rows]
            return tuple(tuple(int(v) for v in row) for row in rows)

        def _grid_cpu_tensor_from_key(grid_key, like_tensor: torch.Tensor | None = None):
            dtype = like_tensor.dtype if torch.is_tensor(like_tensor) else torch.long
            return torch.tensor(grid_key, dtype=dtype)

        def _build_single_image_segment_info(
            input_len: int,
            token_types: list[int],
            grid_tuple: tuple[int, int, int] | None,
            active_tokens: int | None = None,
        ):
            if active_tokens is not None and active_tokens != input_len:
                return None
            if grid_tuple is None:
                return None
            if any(int(v) == 2 for v in token_types):
                return None

            change_positions = [idx + 1 for idx in range(len(token_types) - 1) if token_types[idx + 1] != token_types[idx]]
            if len(change_positions) != 2:
                return None

            image_start, image_end = change_positions
            if not (
                token_types[0] == 0
                and token_types[image_start] == 1
                and image_end < len(token_types)
                and token_types[image_end] == 0
            ):
                return None

            image_token_count = image_end - image_start
            expected_image_tokens = (grid_tuple[0] * grid_tuple[1] * grid_tuple[2]) // (spatial_merge_size**2)
            if image_token_count != expected_image_tokens:
                return None

            image_advance = max(grid_tuple[1], grid_tuple[2]) // spatial_merge_size
            return {
                "image_start": image_start,
                "image_end": image_end,
                "image_token_count": image_token_count,
                "prefix_len": image_start,
                "suffix_len": input_len - image_end,
                "grid_tuple": grid_tuple,
                "spatial_merge_size": spatial_merge_size,
                "image_advance": image_advance,
                "rope_delta_value": image_advance - image_token_count,
            }

        def _extract_prefill_host_meta(batch):
            if not hasattr(batch, "get"):
                return None

            image_grid_thw = batch.get("image_grid_thw")
            grid_key = None
            if torch.is_tensor(image_grid_thw):
                grid_key = _grid_key_from_rows(image_grid_thw.tolist())

            input_ids = batch.get("input_ids")
            mm_token_type_ids = batch.get("mm_token_type_ids")
            video_grid_thw = batch.get("video_grid_thw")
            attention_mask = batch.get("attention_mask")
            segment = None
            can_use = (
                torch.is_tensor(input_ids)
                and torch.is_tensor(mm_token_type_ids)
                and torch.is_tensor(image_grid_thw)
                and video_grid_thw is None
                and input_ids.ndim == 2
                and input_ids.shape[0] == 1
                and mm_token_type_ids.shape == input_ids.shape
                and image_grid_thw.ndim == 2
                and image_grid_thw.shape[0] == 1
            )
            if can_use:
                active_tokens = None
                if attention_mask is not None:
                    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                        can_use = False
                    else:
                        active_tokens = int(attention_mask[0].sum().item())
                if can_use:
                    segment = _build_single_image_segment_info(
                        input_len=int(input_ids.shape[1]),
                        token_types=[int(v) for v in mm_token_type_ids[0].tolist()],
                        grid_tuple=grid_key[0] if grid_key else None,
                        active_tokens=active_tokens,
                    )

            return {
                "grid_key": grid_key,
                "segment": segment,
            }

        def _register_prefill_batch_meta(batch, meta):
            if not meta or not hasattr(batch, "get"):
                return
            image_grid_thw = batch.get("image_grid_thw")
            grid_key = meta.get("grid_key")
            if grid_key and torch.is_tensor(image_grid_thw):
                _cache_put(prefill_grid_key_by_ptr, int(image_grid_thw.data_ptr()), grid_key)

            segment = meta.get("segment")
            segment_key = _segment_ptr_key(
                input_ids=batch.get("input_ids"),
                mm_token_type_ids=batch.get("mm_token_type_ids"),
                image_grid_thw=image_grid_thw,
                video_grid_thw=batch.get("video_grid_thw"),
                attention_mask=batch.get("attention_mask"),
            )
            if segment is not None and segment_key is not None:
                _cache_put(prefill_segment_info_by_ptr, segment_key, segment)

        orig_apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if callable(orig_apply_chat_template):
            def _apply_chat_template_with_prefill_meta(this_processor, *args, **kwargs):
                batch = orig_apply_chat_template(*args, **kwargs)
                if not hasattr(batch, "to"):
                    return batch
                if visual_dtype is not None and hasattr(batch, "data"):
                    for name in ("pixel_values", "pixel_values_videos"):
                        value = batch.data.get(name)
                        if torch.is_tensor(value) and value.device.type == "cpu" and value.dtype != visual_dtype:
                            batch.data[name] = value.to(dtype=visual_dtype)
                batch._prefill_host_meta = _extract_prefill_host_meta(batch) or {}
                orig_to = batch.to

                def _to_with_prefill_meta(this_batch, *to_args, **to_kwargs):
                    result = orig_to(*to_args, **to_kwargs)
                    _register_prefill_batch_meta(this_batch, getattr(this_batch, "_prefill_host_meta", None))
                    return result

                batch.to = MethodType(_to_with_prefill_meta, batch)
                return batch

            self._processor.apply_chat_template = MethodType(_apply_chat_template_with_prefill_meta, self._processor)
            self._optimizations_applied.append("prefill:input_meta_cache")

        def _grid_key(grid_thw):
            if torch.is_tensor(grid_thw):
                cached = prefill_grid_key_by_ptr.get(_tensor_ptr(grid_thw))
                if cached is not None:
                    return cached
                rows = grid_thw.detach().cpu().tolist()
            else:
                rows = grid_thw
            return _grid_key_from_rows(rows)

        def _single_image_segment_info(
            input_ids: torch.Tensor | None,
            mm_token_type_ids: torch.Tensor | None,
            image_grid_thw: torch.Tensor | None,
            video_grid_thw: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
        ):
            can_use = (
                input_ids is not None
                and mm_token_type_ids is not None
                and image_grid_thw is not None
                and video_grid_thw is None
                and input_ids.ndim == 2
                and input_ids.shape[0] == 1
                and mm_token_type_ids.shape == input_ids.shape
                and image_grid_thw.ndim == 2
                and image_grid_thw.shape[0] == 1
            )
            if not can_use:
                return None

            segment_key = _segment_ptr_key(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            if segment_key is not None:
                cached = prefill_segment_info_by_ptr.get(segment_key)
                if cached is not None:
                    return cached

            active_tokens = None
            if attention_mask is not None:
                if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                    return None
                active_tokens = int(attention_mask[0].detach().cpu().sum().item())

            grid_key = _grid_key(image_grid_thw)
            segment = _build_single_image_segment_info(
                input_len=int(input_ids.shape[1]),
                token_types=[int(v) for v in mm_token_type_ids[0].detach().cpu().tolist()],
                grid_tuple=grid_key[0] if grid_key else None,
                active_tokens=active_tokens,
            )
            if segment is not None and segment_key is not None:
                _cache_put(prefill_segment_info_by_ptr, segment_key, segment)
            return segment

        if _ENABLE_PREFILL_VISION_GRID_CACHE:
            if visual is not None:
                from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                    ALL_ATTENTION_FUNCTIONS,
                    BaseModelOutputWithDeepstackFeatures,
                    apply_rotary_pos_emb_vision,
                    eager_attention_forward,
                    is_flash_attention_requested,
                )

                fast_pos_cache = {}
                rot_pos_cache = {}
                position_embeddings_cache = {}
                cu_seqlens_cache = {}
                vision_attn_split_sizes_cache = {}
                vision_pooler_split_sizes_cache = {}
                orig_fast_pos = visual.fast_pos_embed_interpolate
                orig_rot_pos = visual.rot_pos_emb
                orig_visual_forward = visual.forward
                deepstack_merger_index = {
                    int(layer_num): idx for idx, layer_num in enumerate(getattr(visual, "deepstack_visual_indexes", ()))
                }
                original_vision_attn_forwards = {}

                if _ENABLE_PREFILL_PATCH_EMBED_LINEAR:
                    patch_embed = getattr(visual, "patch_embed", None)
                    if patch_embed is not None and hasattr(patch_embed, "proj"):
                        orig_patch_embed_forward = patch_embed.forward
                        linear_weight = patch_embed.proj.weight.flatten(1)
                        linear_bias = patch_embed.proj.bias
                        expected_in_features = int(linear_weight.shape[1])

                        def _patch_embed_linear_forward(this_patch_embed, hidden_states):
                            if hidden_states is None or hidden_states.ndim != 2 or hidden_states.shape[-1] != expected_in_features:
                                return orig_patch_embed_forward(hidden_states)
                            return F.linear(hidden_states.to(dtype=linear_weight.dtype), linear_weight, linear_bias)

                        patch_embed.forward = MethodType(_patch_embed_linear_forward, patch_embed)
                        patch_embed._prefill_linear_weight = linear_weight
                        self._optimizations_applied.append("prefill:patch_embed_linear")

                def _cached_vision_attn_split_sizes(grid_thw):
                    key = _grid_key(grid_thw)
                    cached = vision_attn_split_sizes_cache.get(key)
                    if cached is not None:
                        return cached
                    split_sizes = tuple(int(h) * int(w) for t, h, w in key for _ in range(int(t)))
                    return _cache_put(vision_attn_split_sizes_cache, key, split_sizes)

                def _cached_vision_pooler_split_sizes(grid_thw):
                    key = _grid_key(grid_thw)
                    cached = vision_pooler_split_sizes_cache.get(key)
                    if cached is not None:
                        return cached
                    split_sizes = tuple(
                        (int(t) * int(h) * int(w)) // (spatial_merge_size**2)
                        for t, h, w in key
                    )
                    return _cache_put(vision_pooler_split_sizes_cache, key, split_sizes)

                def _fast_pos_embed_interpolate_cached(this_visual, grid_thw):
                    key = _grid_key(grid_thw)
                    cached = fast_pos_cache.get(key)
                    if cached is not None:
                        return cached
                    return _cache_put(
                        fast_pos_cache,
                        key,
                        orig_fast_pos(_grid_cpu_tensor_from_key(key, grid_thw)),
                    )

                def _rot_pos_emb_cached(this_visual, grid_thw):
                    key = _grid_key(grid_thw)
                    cached = rot_pos_cache.get(key)
                    if cached is not None:
                        return cached
                    return _cache_put(
                        rot_pos_cache,
                        key,
                        orig_rot_pos(_grid_cpu_tensor_from_key(key, grid_thw)),
                    )

                def _cached_vision_geometry(this_visual, grid_thw):
                    key = _grid_key(grid_thw)
                    position_embeddings = position_embeddings_cache.get(key)
                    cu_seqlens = cu_seqlens_cache.get(key)
                    if position_embeddings is None:
                        rotary_pos_emb = this_visual.rot_pos_emb(grid_thw)
                        seq_len = int(rotary_pos_emb.shape[0])
                        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
                        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                        position_embeddings = _cache_put(position_embeddings_cache, key, (emb.cos(), emb.sin()))
                    if cu_seqlens is None:
                        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                            dim=0,
                            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                        )
                        cu_seqlens = _cache_put(cu_seqlens_cache, key, F.pad(cu_seqlens, (1, 0), value=0))
                    return position_embeddings, cu_seqlens

                def _vision_attn_forward_prefill_split_sizes(
                    this_attn,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb=None,
                    position_embeddings=None,
                    **kwargs,
                ):
                    split_sizes = kwargs.pop("prefill_split_sizes", None)
                    orig_attn_forward = original_vision_attn_forwards.get(id(this_attn))
                    if (
                        split_sizes is None
                        or torch.is_grad_enabled()
                        or is_flash_attention_requested(this_attn.config)
                    ):
                        return orig_attn_forward(
                            hidden_states,
                            cu_seqlens,
                            rotary_pos_emb=rotary_pos_emb,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )

                    seq_length = hidden_states.shape[0]
                    query_states, key_states, value_states = (
                        this_attn.qkv(hidden_states).reshape(seq_length, 3, this_attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                    )
                    cos, sin = position_embeddings
                    fused_rope_result = fused_vision_rope(query_states, key_states, cos, sin)
                    if fused_rope_result is not None:
                        query_states, key_states = fused_rope_result
                    else:
                        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

                    query_states = query_states.transpose(0, 1).unsqueeze(0)
                    key_states = key_states.transpose(0, 1).unsqueeze(0)
                    value_states = value_states.transpose(0, 1).unsqueeze(0)

                    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                        this_attn.config._attn_implementation, eager_attention_forward
                    )
                    splits = [
                        torch.split(tensor, split_sizes, dim=2)
                        for tensor in (query_states, key_states, value_states)
                    ]
                    attn_outputs = [
                        attention_interface(
                            this_attn,
                            q,
                            k,
                            v,
                            attention_mask=None,
                            scaling=this_attn.scaling,
                            dropout=0.0 if not this_attn.training else this_attn.attention_dropout,
                            is_causal=False,
                            **kwargs,
                        )[0]
                        for q, k, v in zip(*splits)
                    ]
                    attn_output = torch.cat(attn_outputs, dim=1)
                    attn_output = attn_output.reshape(seq_length, -1).contiguous()
                    return this_attn.proj(attn_output)

                def _visual_forward_cached(this_visual, hidden_states, grid_thw, **kwargs):
                    if (
                        torch.is_grad_enabled()
                        or kwargs.get("return_dict", True) is False
                        or kwargs.get("output_hidden_states")
                        or kwargs.get("output_attentions")
                    ):
                        return orig_visual_forward(hidden_states, grid_thw, **kwargs)

                    hidden_states = this_visual.patch_embed(hidden_states)

                    pos_embeds = this_visual.fast_pos_embed_interpolate(grid_thw)
                    hidden_states = hidden_states + pos_embeds

                    position_embeddings, cu_seqlens = _cached_vision_geometry(this_visual, grid_thw)
                    attn_split_sizes = _cached_vision_attn_split_sizes(grid_thw)

                    seq_len, _ = hidden_states.size()
                    hidden_states = hidden_states.reshape(seq_len, -1)

                    deepstack_feature_lists = []
                    for layer_num, blk in enumerate(this_visual.blocks):
                        hidden_states = blk(
                            hidden_states,
                            cu_seqlens=cu_seqlens,
                            position_embeddings=position_embeddings,
                            prefill_split_sizes=attn_split_sizes,
                            **kwargs,
                        )
                        merger_idx = deepstack_merger_index.get(layer_num)
                        if merger_idx is not None:
                            deepstack_feature = this_visual.deepstack_merger_list[merger_idx](hidden_states)
                            deepstack_feature_lists.append(deepstack_feature)

                    merged_hidden_states = this_visual.merger(hidden_states)
                    return BaseModelOutputWithDeepstackFeatures(
                        last_hidden_state=hidden_states,
                        pooler_output=merged_hidden_states,
                        deepstack_features=deepstack_feature_lists,
                    )

                visual.fast_pos_embed_interpolate = MethodType(_fast_pos_embed_interpolate_cached, visual)
                visual.rot_pos_emb = MethodType(_rot_pos_emb_cached, visual)
                visual.forward = MethodType(_visual_forward_cached, visual)
                for blk in visual.blocks:
                    original_vision_attn_forwards[id(blk.attn)] = blk.attn.forward
                    blk.attn.forward = MethodType(_vision_attn_forward_prefill_split_sizes, blk.attn)
                    blk.attn._prefill_original_forward = original_vision_attn_forwards[id(blk.attn)]
                orig_get_image_features = model_core.get_image_features

                def _get_image_features_cached(this_model, pixel_values, image_grid_thw=None, **kwargs):
                    return_dict = kwargs.pop("return_dict", True)
                    if return_dict is False or image_grid_thw is None or not torch.is_tensor(image_grid_thw):
                        return orig_get_image_features(pixel_values, image_grid_thw, return_dict=return_dict, **kwargs)
                    pixel_values = pixel_values.type(this_model.visual.dtype)
                    vision_output: BaseModelOutputWithDeepstackFeatures = this_model.visual(
                        pixel_values,
                        grid_thw=image_grid_thw,
                        return_dict=True,
                        **kwargs,
                    )
                    image_embeds = vision_output.pooler_output
                    split_sizes = _cached_vision_pooler_split_sizes(image_grid_thw)
                    vision_output.pooler_output = torch.split(image_embeds, split_sizes)
                    return vision_output

                model_core.get_image_features = MethodType(_get_image_features_cached, model_core)
                visual._prefill_fast_pos_cache = fast_pos_cache
                visual._prefill_rot_pos_cache = rot_pos_cache
                visual._prefill_position_embeddings_cache = position_embeddings_cache
                visual._prefill_cu_seqlens_cache = cu_seqlens_cache
                visual._prefill_attn_split_sizes_cache = vision_attn_split_sizes_cache
                visual._prefill_pooler_split_sizes_cache = vision_pooler_split_sizes_cache
                self._optimizations_applied.append("prefill:vision_grid_cache")

        if _ENABLE_PREFILL_MM_SLICE_FASTPATH:
            language_model = getattr(model_core, "language_model", None)
            if language_model is not None:
                from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                    BaseModelOutputWithPast,
                    Qwen3VLModelOutputWithPast,
                )

                orig_model_forward = model_core.forward
                orig_deepstack_process = language_model._deepstack_process
                orig_language_model_forward = language_model.forward
                prefill_text_cache_position_cache = {}
                prefill_text_rotary_cache = {}
                def _cached_prefill_cache_position(device: torch.device, seq_len: int):
                    key = (str(device), int(seq_len))
                    cached = prefill_text_cache_position_cache.get(key)
                    if cached is not None:
                        return cached
                    return _cache_put(
                        prefill_text_cache_position_cache,
                        key,
                        torch.arange(seq_len, dtype=torch.long, device=device),
                    )

                def _cached_prefill_rotary_embeddings(this_text_model, hidden_states, position_ids):
                    if not torch.is_tensor(position_ids) or hidden_states.device.type != "cuda":
                        return this_text_model.rotary_emb(hidden_states, position_ids)
                    key = (
                        str(hidden_states.device),
                        str(hidden_states.dtype),
                        tuple(int(v) for v in position_ids.shape),
                        int(position_ids.data_ptr()),
                    )
                    cached = prefill_text_rotary_cache.get(key)
                    if cached is not None:
                        return cached
                    return _cache_put(
                        prefill_text_rotary_cache,
                        key,
                        this_text_model.rotary_emb(hidden_states, position_ids),
                    )

                def _deepstack_process_slice_fastpath(this_text_model, hidden_states, visual_pos_masks, visual_embeds):
                    token_range = getattr(this_text_model, "_prefill_visual_token_range", None)
                    if (
                        token_range is not None
                        and hidden_states.ndim == 3
                        and hidden_states.shape[0] == 1
                        and visual_embeds is not None
                    ):
                        start, end = token_range
                        if end > start and visual_embeds.ndim == 2 and visual_embeds.shape[0] == (end - start):
                            if visual_embeds.device != hidden_states.device or visual_embeds.dtype != hidden_states.dtype:
                                visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
                            hidden_states = hidden_states.clone()
                            hidden_states[:, start:end, :].add_(visual_embeds.unsqueeze(0))
                            return hidden_states
                    return orig_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)

                def _language_model_forward_prefill_fastpath(
                    this_text_model,
                    input_ids: torch.LongTensor | None = None,
                    attention_mask: torch.Tensor | None = None,
                    position_ids: torch.LongTensor | None = None,
                    past_key_values=None,
                    inputs_embeds: torch.FloatTensor | None = None,
                    use_cache: bool | None = None,
                    cache_position: torch.LongTensor | None = None,
                    visual_pos_masks: torch.Tensor | None = None,
                    deepstack_visual_embeds: list[torch.Tensor] | None = None,
                    **kwargs,
                ):
                    fastpath_state = getattr(this_text_model, "_prefill_text_fastpath_state", None)
                    can_use_fastpath = (
                        _ENABLE_PREFILL_TEXT_FASTPATH
                        and fastpath_state is not None
                        and not torch.is_grad_enabled()
                        and input_ids is None
                        and inputs_embeds is not None
                        and isinstance(past_key_values, StaticCache)
                        and cache_position is None
                        and inputs_embeds.ndim == 3
                        and inputs_embeds.shape[0] == 1
                        and torch.is_tensor(position_ids)
                        and position_ids.ndim == 3
                        and position_ids.shape[-1] == inputs_embeds.shape[1]
                        and fastpath_state.get("full_prompt_mask", False)
                    )
                    if not can_use_fastpath:
                        return orig_language_model_forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            visual_pos_masks=visual_pos_masks,
                            deepstack_visual_embeds=deepstack_visual_embeds,
                            **kwargs,
                        )

                    hidden_states = inputs_embeds
                    cache_position = _cached_prefill_cache_position(hidden_states.device, hidden_states.shape[1])

                    if position_ids.ndim == 2:
                        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

                    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                        text_position_ids = position_ids[0]
                        rotary_position_ids = position_ids[1:]
                    else:
                        text_position_ids = None
                        rotary_position_ids = position_ids

                    position_embeddings = _cached_prefill_rotary_embeddings(
                        this_text_model,
                        hidden_states,
                        rotary_position_ids,
                    )

                    for layer_idx, decoder_layer in enumerate(this_text_model.layers):
                        hidden_states = decoder_layer(
                            hidden_states,
                            attention_mask=None,
                            position_ids=text_position_ids,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            prefill_static_cache_no_mask=True,
                            **kwargs,
                        )

                        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                            hidden_states = this_text_model._deepstack_process(
                                hidden_states,
                                visual_pos_masks,
                                deepstack_visual_embeds[layer_idx],
                            )

                    hidden_states = this_text_model.norm(hidden_states)
                    return BaseModelOutputWithPast(
                        last_hidden_state=hidden_states,
                        past_key_values=past_key_values,
                    )

                def _model_forward_slice_fastpath(
                    this_model,
                    input_ids: torch.LongTensor = None,
                    attention_mask: torch.Tensor | None = None,
                    position_ids: torch.LongTensor | None = None,
                    past_key_values=None,
                    inputs_embeds: torch.FloatTensor | None = None,
                    pixel_values: torch.Tensor | None = None,
                    pixel_values_videos: torch.FloatTensor | None = None,
                    image_grid_thw: torch.LongTensor | None = None,
                    video_grid_thw: torch.LongTensor | None = None,
                    mm_token_type_ids: torch.IntTensor | None = None,
                    cache_position: torch.LongTensor | None = None,
                    **kwargs,
                ):
                    segment = _single_image_segment_info(
                        input_ids=input_ids,
                        mm_token_type_ids=mm_token_type_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask,
                    )
                    if (
                        torch.is_grad_enabled()
                        or kwargs.get("return_dict", True) is False
                        or kwargs.get("output_hidden_states")
                        or kwargs.get("output_attentions")
                        or segment is None
                        or pixel_values is None
                        or pixel_values_videos is not None
                        or inputs_embeds is not None
                    ):
                        return orig_model_forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds,
                            pixel_values=pixel_values,
                            pixel_values_videos=pixel_values_videos,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            mm_token_type_ids=mm_token_type_ids,
                            cache_position=cache_position,
                            **kwargs,
                        )

                    # Compute position_ids BEFORE vision encoder to avoid implicit
                    # cudaStreamSynchronize.  HF's get_rope_index() calls
                    # input_token_type.tolist() which forces a GPU→CPU sync.
                    # Moving this ahead of get_image_features() lets the sync
                    # happen before ~700 vision kernels are queued, eliminating
                    # a ~3.5 ms stall between vision and text decoder phases.
                    if position_ids is None:
                        position_ids = this_model.compute_3d_position_ids(
                            input_ids=input_ids,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            inputs_embeds=None,  # prefill branch doesn't use it
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            mm_token_type_ids=mm_token_type_ids,
                        )

                    inputs_embeds = this_model.get_input_embeddings()(input_ids)
                    image_outputs = this_model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
                    pooler_output = image_outputs.pooler_output
                    if isinstance(pooler_output, (tuple, list)):
                        if len(pooler_output) != 1:
                            return orig_model_forward(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=None,
                                pixel_values=pixel_values,
                                pixel_values_videos=pixel_values_videos,
                                image_grid_thw=image_grid_thw,
                                video_grid_thw=video_grid_thw,
                                mm_token_type_ids=mm_token_type_ids,
                                cache_position=cache_position,
                                **kwargs,
                            )
                        image_embeds = pooler_output[0]
                    else:
                        image_embeds = pooler_output

                    if image_embeds.ndim != 2 or image_embeds.shape[0] != segment["image_token_count"]:
                        return orig_model_forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            inputs_embeds=None,
                            pixel_values=pixel_values,
                            pixel_values_videos=pixel_values_videos,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            mm_token_type_ids=mm_token_type_ids,
                            cache_position=cache_position,
                            **kwargs,
                        )

                    if image_embeds.device != inputs_embeds.device or image_embeds.dtype != inputs_embeds.dtype:
                        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                    inputs_embeds = inputs_embeds.clone()
                    inputs_embeds[:, segment["image_start"] : segment["image_end"], :].copy_(image_embeds.unsqueeze(0))

                    previous_range = getattr(language_model, "_prefill_visual_token_range", None)
                    previous_text_fastpath_state = getattr(language_model, "_prefill_text_fastpath_state", None)
                    language_model._prefill_visual_token_range = (
                        segment["image_start"],
                        segment["image_end"],
                    )
                    language_model._prefill_text_fastpath_state = {
                        "full_prompt_mask": True,
                    }
                    try:
                        outputs = language_model(
                            input_ids=None,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds,
                            cache_position=cache_position,
                            visual_pos_masks=None,
                            deepstack_visual_embeds=image_outputs.deepstack_features,
                            **kwargs,
                        )
                    finally:
                        if previous_range is None:
                            if hasattr(language_model, "_prefill_visual_token_range"):
                                delattr(language_model, "_prefill_visual_token_range")
                        else:
                            language_model._prefill_visual_token_range = previous_range
                        if previous_text_fastpath_state is None:
                            if hasattr(language_model, "_prefill_text_fastpath_state"):
                                delattr(language_model, "_prefill_text_fastpath_state")
                        else:
                            language_model._prefill_text_fastpath_state = previous_text_fastpath_state

                    return Qwen3VLModelOutputWithPast(
                        **outputs,
                        rope_deltas=this_model.rope_deltas,
                    )

                language_model._deepstack_process = MethodType(_deepstack_process_slice_fastpath, language_model)
                language_model.forward = MethodType(_language_model_forward_prefill_fastpath, language_model)
                model_core.forward = MethodType(_model_forward_slice_fastpath, model_core)
                self._optimizations_applied.append("prefill:mm_slice_fastpath")
                if _ENABLE_PREFILL_TEXT_FASTPATH:
                    language_model._prefill_cache_position_cache = prefill_text_cache_position_cache
                    language_model._prefill_rotary_cache = prefill_text_rotary_cache
                    self._optimizations_applied.append("prefill:text_fastpath")
                if _ENABLE_TRITON_PREFILL_FUSED_ADD_RMSNORM:
                    self._optimizations_applied.append("prefill:fused_add_rmsnorm")
                if _ENABLE_TRITON_PREFILL_FUSED_SILU_MUL:
                    self._optimizations_applied.append("prefill:fused_silu_mul")

        if _ENABLE_PREFILL_ROPE_FASTPATH:
            rope_position_cache = {}
            rope_output_cache = {}
            orig_get_rope_index = model_core.get_rope_index
            orig_get_vision_position_ids = model_core.get_vision_position_ids

            def _get_rope_index_fastpath(
                this_model,
                input_ids: torch.LongTensor,
                mm_token_type_ids: torch.IntTensor,
                image_grid_thw: torch.LongTensor | None = None,
                video_grid_thw: torch.LongTensor | None = None,
                attention_mask: torch.Tensor | None = None,
                **kwargs,
            ):
                segment = _single_image_segment_info(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                if segment is None:
                    return orig_get_rope_index(
                        input_ids,
                        mm_token_type_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
                prefix_len = segment["prefix_len"]
                suffix_len = segment["suffix_len"]
                grid_tuple = segment["grid_tuple"]
                spatial_merge_size = segment["spatial_merge_size"]
                device_key = str(input_ids.device)
                rope_key = (device_key, grid_tuple, prefix_len, suffix_len, int(input_ids.shape[1]))
                cached = rope_output_cache.get(rope_key)
                if cached is not None:
                    return cached

                vision_key = (device_key, grid_tuple, prefix_len)
                vision_positions = rope_position_cache.get(vision_key)
                if vision_positions is None:
                    vision_positions = orig_get_vision_position_ids(
                        prefix_len,
                        torch.tensor(grid_tuple, dtype=image_grid_thw.dtype),
                        1,
                        spatial_merge_size,
                        device=input_ids.device,
                    )
                    _cache_put(rope_position_cache, vision_key, vision_positions)

                prefix_positions = (
                    torch.arange(prefix_len, device=input_ids.device, dtype=input_ids.dtype).view(1, -1).expand(3, -1)
                )
                image_advance = segment["image_advance"]
                suffix_positions = (
                    torch.arange(suffix_len, device=input_ids.device, dtype=input_ids.dtype).view(1, -1).expand(3, -1)
                    + (prefix_len + image_advance)
                )
                llm_positions = torch.cat([prefix_positions, vision_positions, suffix_positions], dim=1).reshape(3, -1)
                rope_deltas = torch.tensor(
                    [[segment["rope_delta_value"]]],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                result = (llm_positions.view(3, 1, -1), rope_deltas)
                return _cache_put(rope_output_cache, rope_key, result)

            model_core.get_rope_index = MethodType(_get_rope_index_fastpath, model_core)
            model_core._prefill_rope_position_cache = rope_position_cache
            model_core._prefill_rope_output_cache = rope_output_cache
            self._optimizations_applied.append("prefill:rope_fastpath")

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._wrapped_model

    @property
    def device(self):
        return self._device

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 128
    ) -> Dict:
        """Generate answer (for debugging, benchmark uses model.generate directly)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._wrapped_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]

        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return {
            "text": text,
            "token_count": len(generated_ids)
        }
