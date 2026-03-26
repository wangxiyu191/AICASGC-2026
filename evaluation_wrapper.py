"""
AICAS 2026 - Participant Core Modification File

Optimized VLMModel for Qwen3-VL-2B-Instruct inference.
Phase 1: flash_attn + CUDA config + no accelerate dispatch overhead.
Phase 2: torch.compile on language_model (DISABLED - recompilation issues).
Phase 3: Custom decode loop with StaticCache for throughput optimization.
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
from transformers.cache_utils import StaticCache

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class ModelWrapper:
    """Transparent wrapper around the HF model with custom decode loop."""

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
        """Generate with custom decode loop using StaticCache for throughput optimization."""
        self._call_count += 1

        # Extract generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        use_cache = kwargs.get("use_cache", True)

        # Phase 3: Custom decode loop with StaticCache (DISABLED - under development)
        # Issue: Performance regression compared to original generate
        # TODO: Need proper cache_position handling and attention_mask update
        # if use_cache and not do_sample and max_new_tokens <= self.MAX_TOKENS_THRESHOLD:
        #     try:
        #         return self._generate_with_static_cache(*args, **kwargs)
        #     except Exception as e:
        #         print(f"[ModelWrapper] Custom decode loop failed: {e}, falling back to original")
        #         return self._model.generate(*args, **kwargs)
        return self._model.generate(*args, **kwargs)

    def _generate_with_static_cache(self, *args, **kwargs):
        """Custom generation with StaticCache to eliminate CPU overhead.

        Key optimizations:
        1. Use StaticCache for pre-allocated KV cache (no dynamic allocation)
        2. Manual decode loop without transformers generate overhead
        3. Avoid .item() sync points by using tensor operations
        """
        # Get inputs from args/kwargs
        if args:
            input_ids = args[0]
        else:
            input_ids = kwargs.get('input_ids')

        attention_mask = kwargs.get('attention_mask')
        pixel_values = kwargs.get('pixel_values')
        image_grid_thw = kwargs.get('image_grid_thw')
        max_new_tokens = kwargs.get('max_new_tokens', 128)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Prepare inputs dict for first forward (prefill)
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'use_cache': True,
        }
        if pixel_values is not None:
            model_inputs['pixel_values'] = pixel_values
        if image_grid_thw is not None:
            model_inputs['image_grid_thw'] = image_grid_thw

        # Create StaticCache for pre-allocated KV storage
        # Total length = input length + max_new_tokens
        max_cache_len = seq_len + max_new_tokens
        # Use get_text_config(decoder=True) as in transformers generate
        text_config = self._model.config.get_text_config(decoder=True)
        past_key_values = StaticCache(
            config=text_config,
            max_cache_len=max_cache_len,
            batch_size=batch_size,
            device=device,
            dtype=self._model.dtype,
        )
        model_inputs['past_key_values'] = past_key_values

        # Prefill phase: process full input
        with torch.no_grad():
            outputs = self._model(**model_inputs)

        # Get next token (greedy decoding)
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Collect generated tokens
        generated_tokens = [next_tokens]

        # Decode phase: iterative generation with StaticCache
        for _ in range(max_new_tokens - 1):
            # Prepare inputs for next token (only the new token)
            model_inputs = {
                'input_ids': next_tokens,
                'past_key_values': past_key_values,
                'use_cache': True,
                'cache_position': torch.tensor([past_key_values.get_seq_length() - 1], device=device),
            }

            with torch.no_grad():
                outputs = self._model(**model_inputs)

            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check for EOS token
            if hasattr(self._model.config, 'eos_token_id') and self._model.config.eos_token_id is not None:
                if (next_tokens == self._model.config.eos_token_id).all():
                    break

            generated_tokens.append(next_tokens)

        # Concatenate all tokens: input + generated
        all_generated = torch.cat(generated_tokens, dim=1)
        output_ids = torch.cat([input_ids, all_generated], dim=1)

        return output_ids

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
        # self._apply_torch_compile()

        # Phase 3: Custom decode loop with StaticCache (DISABLED - under development)
        # Code implemented in ModelWrapper._generate_with_static_cache() but not enabled
        # due to performance regression. Needs further debugging.
        # self._optimizations_applied.append("static_cache_decode")

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
