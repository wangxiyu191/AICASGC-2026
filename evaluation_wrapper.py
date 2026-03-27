"""
AICAS 2026 - Participant Core Modification File

Optimized VLMModel for Qwen3-VL-2B-Instruct inference.
Optimizations:
- SDPA attention backend (auto-selects flash/mem_efficient/math per call)
- Fused decoder layers: Add+RMSNorm, QKV proj, gate_up proj, SwiGLU
"""
from typing import Dict
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from optimized_layer import replace_decoder_layers


class ModelWrapper:
    """Transparent wrapper around the HF model with optional tracing."""

    WARMUP_CALLS = 10
    MAX_TOKENS_THRESHOLD = 128

    def __init__(self, model):
        self._model = model
        self._enable_trace = os.environ.get("ENABLE_VIZTRACE", "0") == "1"
        self._call_count = 0
        self._trace_count = 0
        if self._enable_trace:
            from viztracer import VizTracer  # noqa: F401
            self._trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
            os.makedirs(self._trace_dir, exist_ok=True)
            print(f"[ModelWrapper] Trace enabled, output: {self._trace_dir}")

    def _should_trace(self, kwargs):
        if self._call_count <= self.WARMUP_CALLS:
            return False
        if kwargs.get("max_new_tokens", 0) > self.MAX_TOKENS_THRESHOLD:
            return False
        return True

    def generate(self, *args, **kwargs):
        self._call_count += 1
        if not self._enable_trace or not self._should_trace(kwargs):
            return self._model.generate(*args, **kwargs)

        self._trace_count += 1
        from viztracer import VizTracer
        trace_file = os.path.join(self._trace_dir, f"generate_{self._trace_count}.json")
        tracer = VizTracer(log_torch=True, output_file=trace_file, max_stack_depth=15, verbose=0)
        tracer.start()
        try:
            result = self._model.generate(*args, **kwargs)
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
