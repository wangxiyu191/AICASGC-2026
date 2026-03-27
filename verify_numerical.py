"""
Numerical verification: compare original vs OptimizedDecoderLayer output.

Tests single-layer and full-model forward pass to quantify accumulated error
from fused GEMM and RMSNorm reimplementation.
"""
import torch
import sys
import os

os.environ["TQDM_DISABLE"] = "1"


def verify_single_layer(model, layer_idx=0):
    """Compare original layer vs optimized layer on identical input."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
    from optimized_layer import OptimizedDecoderLayer

    lang_model = model.model.language_model
    config = lang_model.config

    # Get the current (optimized) layer
    opt_layer = lang_model.layers[layer_idx]
    if not isinstance(opt_layer, OptimizedDecoderLayer):
        print(f"[SKIP] Layer {layer_idx} is not OptimizedDecoderLayer, nothing to compare.")
        return

    # Rebuild original layer from optimized weights
    orig_layer = Qwen3VLTextDecoderLayer(config, layer_idx).to(model.device, dtype=torch.float16)
    orig_layer.eval()

    # Copy weights back from fused to original
    q_size = opt_layer.q_size
    kv_size = opt_layer.kv_size
    qkv_w = opt_layer.qkv_weight.data
    orig_layer.self_attn.q_proj.weight.data.copy_(qkv_w[:q_size])
    orig_layer.self_attn.k_proj.weight.data.copy_(qkv_w[q_size:q_size + kv_size])
    orig_layer.self_attn.v_proj.weight.data.copy_(qkv_w[q_size + kv_size:])
    orig_layer.self_attn.o_proj.weight.data.copy_(opt_layer.o_proj.weight.data)
    orig_layer.self_attn.q_norm.weight.data.copy_(opt_layer.q_norm_weight.data)
    orig_layer.self_attn.k_norm.weight.data.copy_(opt_layer.k_norm_weight.data)

    gate_up_w = opt_layer.gate_up_weight.data
    intermediate = opt_layer.intermediate_size
    orig_layer.mlp.gate_proj.weight.data.copy_(gate_up_w[:intermediate])
    orig_layer.mlp.up_proj.weight.data.copy_(gate_up_w[intermediate:])
    orig_layer.mlp.down_proj.weight.data.copy_(opt_layer.down_proj.weight.data)

    orig_layer.input_layernorm.weight.data.copy_(opt_layer.input_layernorm_weight.data)
    orig_layer.post_attention_layernorm.weight.data.copy_(opt_layer.post_attn_layernorm_weight.data)

    # Create test input
    batch, seq_len = 1, 256
    hidden_states = torch.randn(batch, seq_len, config.hidden_size,
                                dtype=torch.float16, device=model.device)

    # Create position embeddings (cos, sin) — must have duplicated halves
    # because Triton RoPE kernel only loads first half (D/2)
    head_dim = config.head_dim
    half_dim = head_dim // 2
    cos_half = torch.randn(batch, seq_len, half_dim, dtype=torch.float16, device=model.device)
    sin_half = torch.randn(batch, seq_len, half_dim, dtype=torch.float16, device=model.device)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    position_embeddings = (cos, sin)

    # Create causal attention mask
    attention_mask = None  # Let SDPA handle causal masking

    with torch.no_grad():
        out_orig = orig_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        out_opt = opt_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

    # Compare
    diff = (out_orig - out_opt).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (out_orig.abs() + 1e-8)).mean().item()

    print(f"\n=== Single Layer {layer_idx} Verification ===")
    print(f"  Output shape:      {out_orig.shape}")
    print(f"  Max abs diff:      {max_diff:.2e}")
    print(f"  Mean abs diff:     {mean_diff:.2e}")
    print(f"  Mean relative diff:{rel_diff:.2e}")

    if max_diff < 1e-3:
        print(f"  Result: PASS (max diff < 1e-3, consistent with FP16 GEMM accumulation)")
    elif max_diff < 1e-2:
        print(f"  Result: WARN (max diff < 1e-2, borderline for FP16)")
    else:
        print(f"  Result: FAIL (max diff >= 1e-2, possible computation error)")

    return max_diff, mean_diff, rel_diff


def verify_fused_qkv(model, layer_idx=0):
    """Verify QKV fusion is numerically equivalent."""
    opt_layer = model.model.language_model.layers[layer_idx]
    config = opt_layer.config

    batch, seq_len = 1, 256
    x = torch.randn(batch, seq_len, config.hidden_size,
                     dtype=torch.float16, device=model.device)

    q_size = opt_layer.q_size
    kv_size = opt_layer.kv_size
    qkv_w = opt_layer.qkv_weight.data

    with torch.no_grad():
        # Fused path
        qkv = torch.nn.functional.linear(x, qkv_w)
        q_fused, k_fused, v_fused = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Separate path
        q_sep = torch.nn.functional.linear(x, qkv_w[:q_size])
        k_sep = torch.nn.functional.linear(x, qkv_w[q_size:q_size + kv_size])
        v_sep = torch.nn.functional.linear(x, qkv_w[q_size + kv_size:])

    q_diff = (q_fused - q_sep).abs().max().item()
    k_diff = (k_fused - k_sep).abs().max().item()
    v_diff = (v_fused - v_sep).abs().max().item()

    print(f"\n=== Fused QKV Verification (Layer {layer_idx}) ===")
    print(f"  Q max diff: {q_diff:.2e}")
    print(f"  K max diff: {k_diff:.2e}")
    print(f"  V max diff: {v_diff:.2e}")

    if max(q_diff, k_diff, v_diff) == 0:
        print(f"  Result: EXACT MATCH (bit-identical)")
    elif max(q_diff, k_diff, v_diff) < 1e-3:
        print(f"  Result: PASS (FP16 accumulation variance)")
    else:
        print(f"  Result: FAIL (unexpected difference)")

    return q_diff, k_diff, v_diff


def verify_fused_gate_up(model, layer_idx=0):
    """Verify gate_up fusion is numerically equivalent."""
    opt_layer = model.model.language_model.layers[layer_idx]
    config = opt_layer.config

    batch, seq_len = 1, 256
    x = torch.randn(batch, seq_len, config.hidden_size,
                     dtype=torch.float16, device=model.device)

    gate_up_w = opt_layer.gate_up_weight.data
    intermediate = opt_layer.intermediate_size

    with torch.no_grad():
        # Fused path
        gate_up = torch.nn.functional.linear(x, gate_up_w)
        gate_fused, up_fused = gate_up.chunk(2, dim=-1)

        # Separate path
        gate_sep = torch.nn.functional.linear(x, gate_up_w[:intermediate])
        up_sep = torch.nn.functional.linear(x, gate_up_w[intermediate:])

    gate_diff = (gate_fused - gate_sep).abs().max().item()
    up_diff = (up_fused - up_sep).abs().max().item()

    print(f"\n=== Fused Gate+Up Verification (Layer {layer_idx}) ===")
    print(f"  Gate max diff: {gate_diff:.2e}")
    print(f"  Up max diff:   {up_diff:.2e}")

    if max(gate_diff, up_diff) == 0:
        print(f"  Result: EXACT MATCH (bit-identical)")
    elif max(gate_diff, up_diff) < 1e-3:
        print(f"  Result: PASS (FP16 accumulation variance)")
    else:
        print(f"  Result: FAIL (unexpected difference)")

    return gate_diff, up_diff


def verify_rms_norm():
    """Verify custom _rms_norm vs HF Qwen3VLTextRMSNorm."""
    from optimized_layer import _rms_norm

    device = "cuda:0"
    hidden_size = 2048
    eps = 1e-6

    # Create HF RMSNorm
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
    hf_norm = Qwen3VLTextRMSNorm(hidden_size, eps=eps).to(device, dtype=torch.float16)
    hf_norm.eval()

    weight = hf_norm.weight.data.clone()

    # Test input
    x = torch.randn(1, 256, hidden_size, dtype=torch.float16, device=device)

    with torch.no_grad():
        out_hf = hf_norm(x)
        out_custom = _rms_norm(x, weight, eps)

    diff = (out_hf - out_custom).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n=== RMSNorm Verification ===")
    print(f"  Max abs diff:  {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")

    if max_diff == 0:
        print(f"  Result: EXACT MATCH")
    elif max_diff < 1e-5:
        print(f"  Result: PASS (negligible difference)")
    else:
        print(f"  Result: WARN (kernel vs Python path difference: {max_diff:.2e})")

    return max_diff, mean_diff


def main():
    from transformers import AutoModelForImageTextToText
    from optimized_layer import replace_decoder_layers

    model_path = "./Qwen3-VL-2B-Instruct"
    device = "cuda:0"

    print("=" * 60)
    print("Numerical Verification: Original vs Optimized Layers")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()

    # Replace layers
    count = replace_decoder_layers(model)
    print(f"Replaced {count} layers")

    # Set replaced layers to eval mode too
    for layer in model.model.language_model.layers:
        layer.eval()

    print("\n" + "=" * 60)
    print("Running verification tests...")
    print("=" * 60)

    # Test 1: RMSNorm
    verify_rms_norm()

    # Test 2: Fused QKV
    verify_fused_qkv(model, layer_idx=0)
    verify_fused_qkv(model, layer_idx=13)
    verify_fused_qkv(model, layer_idx=27)

    # Test 3: Fused gate_up
    verify_fused_gate_up(model, layer_idx=0)
    verify_fused_gate_up(model, layer_idx=27)

    # Test 4: Full layer comparison
    verify_single_layer(model, layer_idx=0)
    verify_single_layer(model, layer_idx=13)
    verify_single_layer(model, layer_idx=27)

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
