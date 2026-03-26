"""
AICAS 2026 - Participant Core Modification File

Optimized VLMModel for Qwen3-VL-2B-Instruct inference.
Phase 1: flash_attn + CUDA config + no accelerate dispatch overhead.
Phase 2: torch.compile on language_model for operator fusion.
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

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class ModelWrapper:
    """Transparent wrapper around the HF model."""

    # Number of warmup calls to skip before tracing
    WARMUP_CALLS = 10
    # Only trace generate calls with max_new_tokens <= this threshold
    # (skips accuracy verification calls which use max_new_tokens=1024)
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
            print(f"[ModelWrapper] Trace enabled, files will be saved to {self._trace_dir}")
            print(f"[ModelWrapper] Skipping first {self.WARMUP_CALLS} warmup calls and accuracy (max_new_tokens>{self.MAX_TOKENS_THRESHOLD}) calls")

    def _should_trace(self, kwargs):
        """Determine if this generate call should be traced."""
        # Skip warmup phase
        if self._call_count <= self.WARMUP_CALLS:
            return False
        # Skip accuracy verification (max_new_tokens > threshold)
        max_new_tokens = kwargs.get("max_new_tokens", 0)
        if max_new_tokens > self.MAX_TOKENS_THRESHOLD:
            return False
        return True

    def generate(self, *args, **kwargs):
        self._call_count += 1
        if not self._enable_trace or not self._should_trace(kwargs):
            return self._model.generate(*args, **kwargs)

        self._trace_count += 1
        from viztracer import VizTracer
        trace_file = os.path.join(self._trace_dir, f"generate_{self._trace_count}.json")
        tracer = VizTracer(
            log_torch=True,
            output_file=trace_file,
            max_stack_depth=15,
            verbose=0,
        )
        tracer.start()
        try:
            result = self._model.generate(*args, **kwargs)
        finally:
            tracer.stop()
            tracer.save()
            print(f"[ModelWrapper] Trace saved: {trace_file} (call #{self._call_count}, max_new_tokens={kwargs.get('max_new_tokens', 'N/A')})")
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)


class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self._device = device
        self.model_path = model_path
        self._optimizations_applied = []

        import time as _time

        # CUDA backend config: TF32, flash SDP, etc.
        self._configure_cuda()

        # Load processor
        t0 = _time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(model_path)
        print(f"[VLMModel] Processor loaded in {_time.perf_counter()-t0:.2f}s")

        # Load model with flash_attention_2 if available, fallback to sdpa
        # flash_attention_2 on A800 (eval env), sdpa fallback on other GPUs
        attn_impl = "flash_attention_2" if HAS_FLASH_ATTN else "sdpa"
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

        # Phase 2: torch.compile for language model (DISABLED)
        # Issue: KV Cache dynamic initialization causes frequent recompilations
        # Result: TTFT +0.5%, Throughput -34.8% (worse than Phase 1)
        # Root cause: torch._dynamo hit recompile_limit (8) due to dynamic shapes
        # Solution: Need custom decode loop with StaticCache to eliminate recompilation
        # self._apply_torch_compile()

        # Wrap model
        self._wrapped_model = ModelWrapper(self._model)

        print(f"[VLMModel] Optimizations: {', '.join(self._optimizations_applied)}")

    def _configure_cuda(self):
        """CUDA backend configuration for max performance."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.set_float32_matmul_precision('high')
        self._optimizations_applied.append("cuda_config")

    def _optimize_kv_cache(self):
        """KV Cache basic configuration."""
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id'):
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._model.config.eos_token_id
        self._optimizations_applied.append("kv_cache")

    def _apply_torch_compile(self):
        """Apply torch.compile to language model for operator fusion.

        Note: Only compile language_model, NOT vision_model due to:
        1. flash_attn_varlen_func incompatibility with torch.compile
        2. Dynamic shapes in vision encoder cause frequent recompiles
        """
        try:
            import time as _time
            t0 = _time.perf_counter()

            # Qwen3-VL structure: model.model.language_model is the LLM part
            # model structure: Qwen3VLForConditionalGeneration
            #   -> model: Qwen3VLModel
            #      -> visual: Qwen3VLVisionModel (skip compile)
            #      -> language_model: Qwen3VLTextModel (compile this)
            if hasattr(self._model, 'model') and hasattr(self._model.model, 'language_model'):
                print("[VLMModel] Applying torch.compile to language_model...")
                # Use mode='default' with dynamic=True to handle variable sequence lengths
                # Avoid 'reduce-overhead' due to CUDA Graph conflicts with DeepStack
                self._model.model.language_model = torch.compile(
                    self._model.model.language_model,
                    mode="default",
                    dynamic=True,  # Handle variable sequence lengths
                    fullgraph=False,  # Allow graph breaks for complex control flow
                )
                self._optimizations_applied.append("torch_compile")
                print(f"[VLMModel] torch.compile applied in {_time.perf_counter()-t0:.2f}s")
            else:
                print("[VLMModel] Warning: language_model not found, skipping torch.compile")
        except Exception as e:
            print(f"[VLMModel] Warning: torch.compile failed: {e}")
            # Continue without torch.compile - don't break the model loading

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
