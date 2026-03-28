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
from typing import Dict
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import StaticCache

from optimized_layer import replace_decoder_layers


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
            output_ids[:, prompt_length:prompt_length + 1] = next_tokens

            current_length = prompt_length + 1
            current_position = prompt_length
            eos_token_id = self._resolve_eos_token_id(kwargs)
            eos_token_values = self._resolve_eos_token_values(kwargs)

            if graph_runner is not None:
                try:
                    graph_runner.capture_if_needed(
                        rope_deltas=rope_deltas,
                        start_position=current_position,
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
                            current_length=current_length,
                            max_new_tokens=max_new_tokens,
                            graph_bucket_len=graph_bucket_len,
                        )
                    print(f"[GreedyDecodeEngine] CUDA Graph capture failed, fallback to eager decode: {exc}")
                    graph_runner = None

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
            "ENABLE_CUSTOM_DECODE",
            "ENABLE_VIZTRACE",
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

    def prebuild_graph(self):
        if self._graph is not None:
            return

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
            snapshot.append((
                layer.keys.clone(),
                layer.values.clone(),
                layer.cumulative_length.clone(),
            ))
        return snapshot

    def _restore_cache(self, snapshot):
        for layer, (keys, values, cum_len) in zip(self._static_cache.layers, snapshot):
            layer.keys.copy_(keys)
            layer.values.copy_(values)
            layer.cumulative_length.copy_(cum_len)


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
            self._trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
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

        # Wrap model
        self._wrapped_model = ModelWrapper(self._model)

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
