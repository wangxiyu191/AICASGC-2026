"""
AICAS 2026 - Participant Core Modification File

Optimized VLMModel for Qwen3-VL-2B-Instruct inference.
Phase 1: flash_attn + CUDA config + no accelerate dispatch overhead.
"""
from typing import Dict
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class ModelWrapper:
    """Transparent wrapper around the HF model."""

    def __init__(self, model):
        self._model = model

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

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
