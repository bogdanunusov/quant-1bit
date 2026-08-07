import os
import struct
import json
import urllib.request
import torch
import torch.nn.functional as F
import numpy as np
import time
import gc
from safetensors.torch import save_file

repo_id = "deepseek-ai/DeepSeek-V4-Pro"

print("=" * 80)
print("  DEEPSEEK-V4 PRO: PACKED 1-BIT (bool) + КАСТОМНЫЙ FORWARD")
print("  1 бит/элемент | 32× сжатие | Честный подсчёт памяти")
print("=" * 80)

ARCH = {
    "hidden_size": 6144, "num_attention_heads": 48, "num_key_value_heads": 8,
    "intermediate_size": 16384, "moe_intermediate_size": 2048, "num_experts": 256,
    "num_shared_experts": 1, "num_routed_experts": 256, "topk_routed": 6,
    "topk_total": 7, "num_layers": 61, "vocab_size": 129280,
    "max_position_embeddings": 163840, "rope_theta": 10000, "rms_norm_eps": 1e-6,
    "use_mla": True, "q_lora_rank": 1536, "kv_lora_rank": 512,
    "v_head_dim": 128, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
    "compressor_dim": 512, "compressor_gate_dim": 7168,
    "hc_attn_dim": 24, "hc_ffn_dim": 24, "ffn_gate_dim": 384, "head_dim": 128,
}


def get_shard_header(repo_id, shard_filename):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_filename}"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        header_len = struct.unpack("<Q", resp.read())[0]
    req = urllib.request.Request(url, headers={**headers, "Range": f"bytes=8-{7+header_len}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        header = json.loads(resp.read().decode('utf-8'))
    return header, header_len

def download_tensor_by_key(repo_id, shard_filename, header, header_len, tensor_key):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_filename}"
    headers = {"User-Agent": "Mozilla/5.0"}
    info = header[tensor_key]
    begin, end = info["data_offsets"]
    dtype_str = info["dtype"]
    shape = info["shape"]
    abs_begin = 8 + header_len + begin
    abs_end = 8 + header_len + end - 1
    req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={abs_begin}-{abs_end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw_bytes = resp.read()
    if dtype_str == "BF16":
        arr = np.frombuffer(raw_bytes, dtype=np.uint16).copy()
        tensor = torch.from_numpy(arr).view(torch.bfloat16)
    elif dtype_str == "F16":
        arr = np.frombuffer(raw_bytes, dtype=np.float16).copy()
        tensor = torch.from_numpy(arr)
    elif dtype_str == "F32":
        arr = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        tensor = torch.from_numpy(arr)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return tensor.reshape(shape)

def infer_layer_architecture(key, shape):
    key_lower = key.lower()
    if "embed" in key_lower and len(shape) == 2: return "embedding", shape[1], shape[0], "embedding"
    if "lm_head" in key_lower and len(shape) == 2: return "lm_head", shape[1], shape[0], "linear"
    if "norm" in key_lower and len(shape) == 1: return "norm", shape[0], shape[0], "rmsnorm"
    if "attn_sink" in key_lower and len(shape) == 1: return "attn_sink", shape[0], shape[0], "none"
    if "hc_attn_base" in key_lower and len(shape) == 1: return "hc_base", shape[0], shape[0], "none"
    if "hc_attn_scale" in key_lower and len(shape) == 1: return "hc_scale", shape[0], shape[0], "none"
    if "hc_ffn_base" in key_lower and len(shape) == 1: return "hc_base", shape[0], shape[0], "none"
    if "hc_ffn_scale" in key_lower and len(shape) == 1: return "hc_scale", shape[0], shape[0], "none"
    if "hc_attn_fn" in key_lower and len(shape) == 2: return "hc_fn", shape[1], shape[0], "linear"
    if "hc_ffn_fn" in key_lower and len(shape) == 2: return "hc_fn", shape[1], shape[0], "linear"
    if "compressor" in key_lower:
        if "ape" in key_lower and len(shape) == 2: return "compressor_ape", shape[1], shape[0], "linear"
        elif "wgate" in key_lower and len(shape) == 2: return "compressor_gate", shape[1], shape[0], "linear"
        elif "wkv" in key_lower and len(shape) == 2: return "compressor_wkv", shape[1], shape[0], "linear"
        elif len(shape) == 1: return "compressor_norm", shape[0], shape[0], "rmsnorm"
    if any(x in key_lower for x in ["q_proj", "k_proj", "v_proj", "o_proj"]) and len(shape) == 2: return "attention_proj", shape[1], shape[0], "linear"
    if any(x in key_lower for x in ["q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "q_down_proj", "q_up_proj"]) and len(shape) == 2: return "mla_proj", shape[1], shape[0], "linear"
    if "ffn.gate" in key_lower and len(shape) == 2: return "ffn_gate", shape[1], shape[0], "linear"
    if any(x in key_lower for x in ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"]) and len(shape) == 2: return "ffn_linear", shape[1], shape[0], "linear"
    if ("gate" in key_lower or "router" in key_lower) and len(shape) == 2: return "router", shape[1], shape[0], "linear"
    if len(shape) == 2: return "generic_linear", shape[1], shape[0], "linear"
    if len(shape) == 1: return "generic_1d", shape[0], shape[0], "none"
    return "generic_nd", shape[-1], shape[0], "none"

def get_adaptive_ratio(layer_type, numel):
    ratios = {"embedding": 0.95, "lm_head": 0.95, "norm": 0.99, "compressor_norm": 0.99, "attention_proj": 0.88, "mla_proj": 0.88, "router": 0.90, "ffn_gate": 0.82, "ffn_linear": 0.82, "compressor_gate": 0.88, "compressor_wkv": 0.88, "compressor_ape": 0.88, "hc_fn": 0.85, "generic_linear": 0.85, "generic_1d": 0.95, "generic_nd": 0.85}
    base = ratios.get(layer_type, 0.85)
    if numel > 10_000_000: base = max(base - 0.05, 0.80)
    return base


class PackedOneBitTensor:
    def __init__(self, packed_bits, scale, shape, numel):
        self.packed_bits = packed_bits
        self.scale = scale
        self.shape = shape
        self.numel = numel

    @staticmethod
    def from_float(W_float, coverage_ratio=1.0):
        device = W_float.device
        numel = W_float.numel()
        shape = W_float.shape

        if len(shape) >= 2 and coverage_ratio < 1.0:
            rows, cols = shape[0], shape[-1]
            block_size = 8
            pad_rows = (block_size - rows % block_size) % block_size
            pad_cols = (block_size - cols % block_size) % block_size
            W_padded = F.pad(W_float, (0, pad_cols, 0, pad_rows))
            nr, nc = W_padded.shape[0], W_padded.shape[-1]
            nrb, ncb = nr // block_size, nc // block_size
            W_blocks = W_padded.reshape(nrb, block_size, ncb, block_size)
            block_energy = W_blocks.abs().sum(dim=(1, 3))
            total_blocks = nrb * ncb
            n_masked = max(1, int(total_blocks * coverage_ratio))
            _, top_idx = torch.topk(block_energy.flatten(), n_masked)
            bmask = torch.zeros(total_blocks, device=device)
            bmask[top_idx] = 1.0
            bmask = bmask.reshape(nrb, ncb)
            emask = bmask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:rows, :cols]
            if len(shape) > 2:
                emask = emask.reshape(shape[-2:])
                for _ in range(len(shape) - 2): emask = emask.unsqueeze(0)
                emask = emask.expand(shape)
        else:
            emask = torch.ones_like(W_float)

        W_masked = W_float * emask
        sign = (W_masked >= 0).bool()
        scale = (W_masked.abs().sum() / (emask.sum() + 1e-8)).item()
        coverage = emask.mean().item()
        sign_flat = sign.flatten().cpu().numpy()
        packed = np.packbits(sign_flat)
        packed_bits = torch.from_numpy(packed).to(torch.uint8)

        return PackedOneBitTensor(packed_bits, scale, shape, numel), coverage

    def memory_bytes(self):
        return self.packed_bits.numel() * 1 + 4

    def memory_mb(self):
        return self.memory_bytes() / (1024**2)

    def to_dense_float32(self, device="cuda"):
        unpacked = np.unpackbits(self.packed_bits.cpu().numpy())
        sign_bool = unpacked[:self.numel]
        sign_float = torch.where(torch.from_numpy(sign_bool).to(device), 1.0, -1.0)
        return (self.scale * sign_float).reshape(self.shape).to(device)


class RealGenerationInputs:
    def __init__(self, repo_id, device="cuda", batch=4, seq=512):
        self.repo_id = repo_id
        self.device = device
        self.batch = batch
        self.seq = seq
        self.embed_weights = None
        self._load_embeddings()
        self._prepare_test_inputs()

    def _load_embeddings(self):
        print("[*] Загрузка реальных эмбеддингов из model-00001-of-00064...")
        h1, h1_len = get_shard_header(self.repo_id, "model-00001-of-00064.safetensors")
        embed_keys = [k for k, v in h1.items() if k != "__metadata__" and "embed" in k.lower()]
        if not embed_keys:
            embed_keys = [k for k, v in h1.items() if k != "__metadata__" and v.get("dtype") in ["F16", "BF16", "F32"] and len(v.get("shape", [])) == 2 and v["shape"][0] == ARCH["vocab_size"]]
        if not embed_keys:
            raise RuntimeError("Не найдены эмбеддинги в шарде 1!")
        embed_key = embed_keys[0]
        embed = download_tensor_by_key(self.repo_id, "model-00001-of-00064.safetensors", h1, h1_len, embed_key)
        self.embed_weights = embed.to(device=self.device, dtype=torch.float32)
        self.vocab_size, self.hidden_size = self.embed_weights.shape
        print(f"[+] Эмбеддинги: {embed_key} -> ({self.vocab_size}, {self.hidden_size})")
        del h1, embed
        torch.cuda.empty_cache()

    def _prepare_test_inputs(self):
        self.token_ids = torch.randint(0, self.vocab_size, (self.batch, self.seq), device=self.device)
        self.embeddings = F.embedding(self.token_ids, self.embed_weights)
        self._cache = {}

    def get_input_for_arch(self, layer_type, in_dim, out_dim):
        cache_key = (layer_type, in_dim, out_dim)
        if cache_key in self._cache:
            return self._cache[cache_key]

        x = self.embeddings

        if layer_type == "embedding":
            result = (self.token_ids, "embedding")
        elif layer_type in ("norm", "compressor_norm"):
            if in_dim != self.hidden_size:
                proj = torch.randn(self.hidden_size, in_dim, device=self.device, dtype=torch.float32) / (self.hidden_size ** 0.5)
                x = torch.matmul(x, proj)
            result = (x, "rmsnorm")
        elif layer_type in ("router", "ffn_gate", "compressor_gate", "compressor_wkv", "compressor_ape", "hc_fn", "attention_proj", "mla_proj", "ffn_linear", "lm_head", "generic_linear"):
            if in_dim != self.hidden_size:
                proj = torch.randn(self.hidden_size, in_dim, device=self.device, dtype=torch.float32) / (self.hidden_size ** 0.5)
                x = torch.matmul(x, proj)
            result = (x, "linear")
        else:
            result = (None, "none")

        self._cache[cache_key] = result
        return result


def measure_unpack_time(packed_w, device="cuda", num_runs=100):
    """Измеряем только время распаковки packed_bits -> dense float32"""
    torch.cuda.synchronize()
    gc.collect()

    # Warmup
    for _ in range(10):
        _ = packed_w.to_dense_float32(device)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_runs):
        w = packed_w.to_dense_float32(device)
    torch.cuda.synchronize()
    elapsed_ms = ((time.time() - t0) / num_runs) * 1000

    # Память распакованных весов
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    w = packed_w.to_dense_float32(device)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    unpack_mem_mb = (mem_after - mem_before) / (1024**2)
    del w
    torch.cuda.empty_cache()

    return elapsed_ms, unpack_mem_mb


def measure_forward_only(fn, w_dense, x_input, num_warmup=10, num_runs=100):
    """Измеряем только forward pass, веса уже dense"""
    device = "cuda"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    gc.collect()

    mem_before = torch.cuda.memory_allocated()

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = fn(x_input, w_dense)

    torch.cuda.synchronize()

    # Измерение
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(num_runs):
            result = fn(x_input, w_dense)
    torch.cuda.synchronize()
    elapsed_ms = ((time.time() - t0) / num_runs) * 1000

    peak_mem = torch.cuda.max_memory_allocated()
    peak_mb = peak_mem / (1024**2)
    forward_mem_mb = (torch.cuda.memory_allocated() - mem_before) / (1024**2)

    del result
    torch.cuda.empty_cache()

    return elapsed_ms, peak_mb, forward_mem_mb


class RealGenerationR2Checker:
    def __init__(self, config, tensor_key, shape, real_inputs):
        self.config = config
        self.tensor_key = tensor_key
        self.shape = shape
        self.real_inputs = real_inputs
        self.layer_type, self.in_dim, self.out_dim, self.op_type = infer_layer_architecture(tensor_key, shape)
        print(f"    [ARCH] {tensor_key} -> type={self.layer_type}, in={self.in_dim}, out={self.out_dim}, op={self.op_type}")

    def compute_all(self, W_orig, packed_w):
        device = "cuda"
        dtype = torch.float32

        if self.op_type == "none":
            W_recon = packed_w.to_dense_float32()
            ss_res = torch.sum((W_orig - W_recon) ** 2)
            ss_tot = torch.sum((W_orig - torch.mean(W_orig)) ** 2)
            r2 = (1.0 - (ss_res / (ss_tot + 1e-8))).item()
            rel_err = (torch.norm(W_orig - W_recon) / (torch.norm(W_orig) + 1e-8)).item()
            return r2, rel_err, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        x_input, forward_type = self.real_inputs.get_input_for_arch(self.layer_type, self.in_dim, self.out_dim)

        def make_fn(ft):
            if ft == "embedding":
                return lambda x, w: F.embedding(x, w)
            elif ft == "rmsnorm":
                return lambda x, w: x * w.view(1, 1, -1) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.config["rms_norm_eps"])
            elif ft == "linear":
                return lambda x, w: F.linear(x, w) if len(w.shape) == 2 else torch.matmul(x, w)
            else:
                return lambda x, w: F.linear(x if x is not None else torch.randn(self.real_inputs.batch, self.real_inputs.seq, self.in_dim, device=device, dtype=dtype), w)

        fn = make_fn(forward_type)

        # === ОРИГИНАЛ ===
        w_orig_dense = W_orig.to(device, dtype=dtype)

        # Forward original
        t_orig_fwd, peak_orig, fwd_mem_orig = measure_forward_only(fn, w_orig_dense, x_input)

        del w_orig_dense
        torch.cuda.empty_cache()

        # === PACKED: ВОССТАНОВЛЕНИЕ ВЕСОВ (ОТДЕЛЬНО) ===
        t_unpack, unpack_mem = measure_unpack_time(packed_w, device)
        print(f"    [UNPACK] Время восстановления: {t_unpack:.3f} ms | Память: {unpack_mem:.4f} MB")

        # === PACKED: FORWARD (УЖЕ ВОССТАНОВЛЕННЫЕ ВЕСА) ===
        w_packed_dense = packed_w.to_dense_float32(device)
        t_packed_fwd, peak_packed, fwd_mem_packed = measure_forward_only(fn, w_packed_dense, x_input)

        del w_packed_dense
        torch.cuda.empty_cache()

        # === R² НА РЕАЛЬНЫХ ВХОДАХ ===
        W_recon = packed_w.to_dense_float32()
        with torch.inference_mode():
            if forward_type == "embedding":
                y_orig = F.embedding(x_input, W_orig.to(device, dtype))
                y_recon = F.embedding(x_input, W_recon)
            elif forward_type == "rmsnorm":
                var = x_input.pow(2).mean(-1, keepdim=True)
                rsqrt = torch.rsqrt(var + self.config["rms_norm_eps"])
                y_orig = x_input * W_orig.to(device, dtype).view(1, 1, -1) * rsqrt
                y_recon = x_input * W_recon.view(1, 1, -1) * rsqrt
            elif forward_type == "linear":
                x = x_input
                if len(W_orig.shape) == 2:
                    y_orig = F.linear(x, W_orig.to(device, dtype))
                    y_recon = F.linear(x, W_recon)
                else:
                    y_orig = torch.matmul(x, W_orig.to(device, dtype))
                    y_recon = torch.matmul(x, W_recon)
            else:
                x = x_input if x_input is not None else torch.randn(self.real_inputs.batch, self.real_inputs.seq, self.in_dim, device=device, dtype=dtype)
                y_orig = F.linear(x, W_orig.to(device, dtype))
                y_recon = F.linear(x, W_recon)

        ss_res = torch.sum((y_orig - y_recon) ** 2)
        ss_tot = torch.sum((y_orig - torch.mean(y_orig)) ** 2)
        r2 = (1.0 - (ss_res / (ss_tot + 1e-8))).item()
        rel_err = (torch.norm(y_orig - y_recon) / (torch.norm(y_orig) + 1e-8)).item()

        token_mse = None
        if self.layer_type in ("embedding", "lm_head"):
            token_mse = torch.mean((y_orig - y_recon) ** 2, dim=-1).mean().item()

        del W_recon, y_orig, y_recon
        torch.cuda.empty_cache()

        # Теоретические размеры
        orig_theoretical_mb = W_orig.numel() * 4 / (1024**2)
        packed_theoretical_mb = packed_w.memory_mb()

        return r2, rel_err, token_mse, peak_orig, peak_packed, fwd_mem_orig, fwd_mem_packed, orig_theoretical_mb, packed_theoretical_mb, t_orig_fwd, t_packed_fwd, t_unpack, unpack_mem


def process_all_tensors_chunked_real(repo_id, shard_filename, header, header_len, real_inputs, chunk_size=3):
    all_keys = [k for k in header.keys() if k != "__metadata__"]
    numeric_keys = []
    for k in all_keys:
        info = header[k]
        dtype = info.get("dtype", "")
        shape = info.get("shape", [])
        if dtype in ["F16", "BF16", "F32"] and len(shape) > 0 and all(s > 0 for s in shape):
            numeric_keys.append(k)
    print(f"\n[+] Всего тензоров: {len(all_keys)} | Числовых: {len(numeric_keys)}")
    results = []
    total = len(numeric_keys)
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_keys = numeric_keys[chunk_start:chunk_end]
        print(f"\n{'='*70}")
        print(f"  ЧАНК {chunk_start//chunk_size + 1}/{(total-1)//chunk_size + 1} | Тензоры {chunk_start+1}-{chunk_end}")
        print(f"{'='*70}")
        for k in chunk_keys:
            try:
                t = download_tensor_by_key(repo_id, shard_filename, header, header_len, k)
                t_gpu = t.to(device="cuda", dtype=torch.float32)
                shape = list(t_gpu.shape)
                numel = t_gpu.numel()
                if numel < 10:
                    print(f"  [SKIP] {k}: слишком мало элементов ({numel})")
                    del t, t_gpu
                    torch.cuda.empty_cache()
                    continue
                layer_type, _, _, _ = infer_layer_architecture(k, shape)
                ratio = get_adaptive_ratio(layer_type, numel)
                print(f"\n  [PROCESS] {k}: shape={shape}, elements={numel}, type={layer_type}, ratio={ratio:.2%}")

                packed_w, coverage = PackedOneBitTensor.from_float(t_gpu, coverage_ratio=ratio)

                checker = RealGenerationR2Checker(ARCH, k, shape, real_inputs)
                r2, rel_err, token_mse, peak_orig, peak_packed, fwd_mem_orig, fwd_mem_packed, orig_theory, packed_theory, t_orig_fwd, t_packed_fwd, t_unpack, unpack_mem = checker.compute_all(t_gpu, packed_w)

                print(f"    -> R² (1-bit vs orig): {r2:.6f}")
                print(f"    -> Relative error: {rel_err:.6f}")
                if token_mse is not None:
                    print(f"    -> Token MSE: {token_mse:.6f}")
                print(f"    -> Scale: {packed_w.scale:.6f} | Coverage: {coverage:.2%}")
                print(f"    -> ВЕСЫ теор.: ориг={orig_theory:.4f} MB | packed={packed_theory:.6f} MB | сжатие={orig_theory/packed_theory:.1f}x")
                print(f"    -> FORWARD ВРЕМЯ: ориг={t_orig_fwd:.3f} ms | packed={t_packed_fwd:.3f} ms | ускорение={t_orig_fwd/t_packed_fwd:.2f}x")
                print(f"    -> FORWARD ПАМЯТЬ: ориг={fwd_mem_orig:.4f} MB | packed={fwd_mem_packed:.4f} MB")
                print(f"    -> PEAK: ориг={peak_orig:.2f} MB | packed={peak_packed:.2f} MB")
                print(f"    -> ВОССТАНОВЛЕНИЕ: {t_unpack:.3f} ms | {unpack_mem:.4f} MB")

                results.append({
                    "key": k, "shape": shape, "ndim": len(shape),
                    "r2": r2, "rel_error": rel_err, "token_mse": token_mse,
                    "scale": packed_w.scale, "coverage": coverage,
                    "orig_theory_mb": orig_theory, "packed_theory_mb": packed_theory,
                    "fwd_mem_orig_mb": fwd_mem_orig, "fwd_mem_packed_mb": fwd_mem_packed,
                    "peak_orig_mb": peak_orig, "peak_packed_mb": peak_packed,
                    "time_orig_fwd_ms": t_orig_fwd, "time_packed_fwd_ms": t_packed_fwd,
                    "time_unpack_ms": t_unpack, "unpack_mem_mb": unpack_mem,
                    "numel": numel,
                    "orig_bytes": header[k]["data_offsets"][1] - header[k]["data_offsets"][0],
                    "layer_type": checker.layer_type,
                })

                del t, t_gpu, packed_w, checker
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"    [VRAM] {allocated:.2f} GB")
            except Exception as e:
                print(f"    [ERROR] {k}: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [ЧАНК завершён] VRAM очищена")
    return results


def save_results(results, output_file="deepseek_v4_pro_packed_1bit.safetensors"):
    save_dict = {}
    for r in results:
        kb = r["key"].replace(".", "_")
        save_dict[f"{kb}_scale"] = torch.tensor([r["scale"]], dtype=torch.float32)
        meta = [r["r2"], r["rel_error"], r["coverage"], float(r["numel"]),
                r["token_mse"] if r["token_mse"] is not None else -1.0,
                r["orig_theory_mb"], r["packed_theory_mb"],
                r["fwd_mem_orig_mb"], r["fwd_mem_packed_mb"],
                r["peak_orig_mb"], r["peak_packed_mb"],
                r["time_orig_fwd_ms"], r["time_packed_fwd_ms"],
                r["time_unpack_ms"], r["unpack_mem_mb"]]
        save_dict[f"{kb}_meta"] = torch.tensor(meta, dtype=torch.float32)
    save_file(save_dict, output_file)
    return os.path.getsize(output_file)


def main():
    shard_filename = "model-00002-of-00064.safetensors"
    print(f"\n[*] Чтение заголовка {shard_filename}...")
    header, header_len = get_shard_header(repo_id, shard_filename)
    print(f"\n[*] Подготовка реальных входных данных (random tokens -> embeddings)...")
    real_inputs = RealGenerationInputs(repo_id, batch=4, seq=512)
    print(f"\n[*] Обработка (packed 1-bit + разделённый forward/unpack)...")
    results = process_all_tensors_chunked_real(repo_id, shard_filename, header, header_len, real_inputs, chunk_size=3)
    if not results:
        print("[!] Нет результатов!")
        return

    print(f"\n{'='*80}")
    print(f"  ИТОГОВАЯ СТАТИСТИКА: {len(results)} тензоров")
    print(f"{'='*80}")

    r2_vals = [r["r2"] for r in results]
    orig_theory = [r["orig_theory_mb"] for r in results]
    packed_theory = [r["packed_theory_mb"] for r in results]
    time_orig_fwd = [r["time_orig_fwd_ms"] for r in results]
    time_packed_fwd = [r["time_packed_fwd_ms"] for r in results]
    time_unpack = [r["time_unpack_ms"] for r in results]

    print(f"\n  R² (1-bit vs оригинал):")
    print(f"    Mean:  {np.mean(r2_vals):.4f} | Median:{np.median(r2_vals):.4f}")
    print(f"    Min:   {np.min(r2_vals):.4f} | Max:   {np.max(r2_vals):.4f}")
    above_90 = sum(1 for v in r2_vals if v >= 0.90)
    above_95 = sum(1 for v in r2_vals if v >= 0.95)
    print(f"    >=0.90: {above_90}/{len(r2_vals)} ({above_90/len(r2_vals)*100:.1f}%)")
    print(f"    >=0.95: {above_95}/{len(r2_vals)} ({above_95/len(r2_vals)*100:.1f}%)")

    print(f"\n  ТЕОРЕТИЧЕСКИЙ РАЗМЕР ВЕСОВ (MB):")
    print(f"    Оригинал:     Mean={np.mean(orig_theory):.4f} | Median={np.median(orig_theory):.4f}")
    print(f"    Packed 1-bit: Mean={np.mean(packed_theory):.6f} | Median={np.median(packed_theory):.6f}")
    print(f"    Сжатие:       {np.mean(orig_theory)/np.mean(packed_theory):.1f}x")

    print(f"\n  FORWARD ВРЕМЯ (ms):")
    print(f"    Оригинал: Mean={np.mean(time_orig_fwd):.3f} | Median={np.median(time_orig_fwd):.3f}")
    print(f"    Packed:   Mean={np.mean(time_packed_fwd):.3f} | Median={np.median(time_packed_fwd):.3f}")
    speedups = [o/p if p > 0 else 0 for o, p in zip(time_orig_fwd, time_packed_fwd)]
    print(f"    Ускорение: Mean={np.mean(speedups):.2f}x | Median={np.median(speedups):.2f}x")

    print(f"\n  ВОССТАНОВЛЕНИЕ ВЕСОВ (unpack, ms):")
    print(f"    Mean:  {np.mean(time_unpack):.3f} | Median:{np.median(time_unpack):.3f}")

    total_orig_mb = sum(r["orig_bytes"] for r in results) / (1024**2)
    total_packed_mb = sum(r["packed_theory_mb"] for r in results)

    print(f"\n{'='*80}")
    print(f"  СЖАТИЕ ВЕСОВ (всего)")
    print(f"{'='*80}")
    print(f"  Оригинал:     {total_orig_mb:.2f} MB")
    print(f"  Packed 1-bit: {total_packed_mb:.4f} MB")
    print(f"  Коэффициент:  {total_orig_mb/total_packed_mb:.1f}x")

    out_file = "deepseek_v4_pro_packed_1bit_results.safetensors"
    saved_bytes = save_results(results, out_file)
    print(f"\n  Сохранено: {out_file} ({saved_bytes/1024:.2f} KB)")

    print(f"\n{'='*80}")
    print(f"  ГОТОВО! Packed 1-bit | Forward отдельно | Unpack отдельно")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
