import os
import struct
import json
import urllib.request
import gc
import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import save_file

repo_id = "deepseek-ai/DeepSeek-V4-Pro"

print("=== COMPLETE DEEPSEEK-V4 CHUNKED STACKING (ALL TENSORS 1D/2D/3D/4D & BF16) ===")

def get_shard_header(repo_id, shard_filename):
    """ Скачивает JSON-заголовок файла и возвращает данные обо ВСЕХ тензорах без исключения """
    url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_filename}"
    headers = {"User-Agent": "Mozilla/5.0"}

    req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        header_len = struct.unpack("<Q", resp.read())[0]

    req = urllib.request.Request(url, headers={**headers, "Range": f"bytes=8-{7+header_len}"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        header = json.loads(resp.read().decode('utf-8'))

    tensors_info = []
    # Карта размеров одного элемента в байтах (ТОЛЬКО BF16)
    dtype_map = {
        "BF16": 2
    }

    for key, info in header.items():
        if key == "__metadata__":
            continue
        shape = info.get("shape", [])
        dtype_str = info.get("dtype", "")

        if dtype_str in dtype_map and len(shape) > 0:
            begin, end = info["data_offsets"]
            abs_begin = 8 + header_len + begin
            tensors_info.append({
                "key": key,
                "shape": shape,
                "dtype": dtype_str,
                "abs_begin": abs_begin,
                "elem_bytes": dtype_map[dtype_str]
            })

    return tensors_info

def download_tensor_chunk(repo_id, shard_filename, abs_begin, elem_bytes, total_cols, row_start, row_end, dtype_str):
    """ Скачивает порцию байт с сервера HF для BF16 """
    url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_filename}"
    headers = {"User-Agent": "Mozilla/5.0"}

    byte_start = abs_begin + (row_start * total_cols * elem_bytes)
    byte_end = abs_begin + (row_end * total_cols * elem_bytes) - 1

    req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={byte_start}-{byte_end}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_bytes = resp.read()

    # Приводим к PyTorch тензору BF16
    arr = np.frombuffer(raw_bytes, dtype=np.uint16).copy()
    tensor = torch.from_numpy(arr).view(torch.bfloat16)

    return tensor.reshape(row_end - row_start, total_cols)



CHUNK_ROWS = 1024  # Фиксированный размер чанка по строкам

for shard_idx in range(1, 65):
    shard_filename = f"model-{shard_idx:05d}-of-00064.safetensors"
    print(f"\n==================== ШАРД {shard_idx}/64: {shard_filename} ====================")

    all_tensors = get_shard_header(repo_id, shard_filename)
    print(f"[*] В шарде обнаружено абсолютно ВСЕХ тензоров для обработки: {len(all_tensors)}")

    shard_results = {}

    for t_idx, tensor_info in enumerate(all_tensors, 1):
        tensor_key = tensor_info["key"]
        shape = tensor_info["shape"]
        dtype_str = tensor_info["dtype"]
        abs_begin = tensor_info["abs_begin"]
        elem_bytes = tensor_info["elem_bytes"]

        # 1. ОБРАБОТКА 1D ТЕНЗОРОВ (Нормализации, скаляры, смещения)
        if len(shape) == 1:
            print(f"\n ---> [{t_idx}/{len(all_tensors)}] 1D-Вектор (без сжатия): {tensor_key} {shape} ({dtype_str})")
            vec_chunk = download_tensor_chunk(repo_id, shard_filename, abs_begin, elem_bytes, shape[0], 0, 1, dtype_str)
            shard_results[tensor_key] = vec_chunk.cpu()
            del vec_chunk
            continue

        # 2. ОБРАБОТКА МНОГОМЕРНЫХ МАТРИЦ (2D, 3D, 4D Эксперты MoE)
        # Виртуально преобразуем тензор любой мерности вида [E, N, M] в 2D-матрицу
        in_features = shape[-1]
        out_features = int(np.prod(shape[:-1]))

        print(f"\n ---> [{t_idx}/{len(all_tensors)}] Многомерная матрица: {tensor_key} {shape} -> [Flatten 2D: {out_features}, {in_features}] ({dtype_str})")

        x_test = torch.randn(1, in_features, device="cuda", dtype=torch.bfloat16)

        # Выгружаем мини-чанк для расчета квантиля
        sample_rows = min(512, out_features)
        sample_chunk = download_tensor_chunk(repo_id, shard_filename, abs_begin, elem_bytes, in_features, 0, sample_rows, dtype_str)
        base_thresh_val = torch.quantile(sample_chunk.abs().float(), 0.30).item()
        del sample_chunk

        packed_masks_list = []
        scalar_layers_accum = np.zeros((5, 2), dtype=np.float32)
        ss_res_total, ss_tot_total = 0.0, 0.0

        # Почанковый прогон матрицы на GPU
        for r_start in range(0, out_features, CHUNK_ROWS):
            r_end = min(r_start + CHUNK_ROWS, out_features)

            W_chunk = download_tensor_chunk(repo_id, shard_filename, abs_begin, elem_bytes, in_features, r_start, r_end, dtype_str)
            W_chunk_gpu = W_chunk.to(device="cuda", dtype=torch.bfloat16)
            del W_chunk

            Y_true_chunk = torch.matmul(x_test, W_chunk_gpu.T)
            nonzero_mask = (torch.abs(W_chunk_gpu) >= base_thresh_val)

            W_current_target = W_chunk_gpu.clone()
            W_reconstructed_chunk = torch.zeros_like(W_chunk_gpu)

            for pass_idx in range(5):
                mask_pos = ((W_current_target > 0) & nonzero_mask)
                mask_neg = ((W_current_target < 0) & nonzero_mask)

                pos_count = torch.sum(mask_pos)
                neg_count = torch.sum(mask_neg)

                s_pos = (torch.sum(W_current_target[mask_pos]) / (pos_count + 1e-8)).item() if pos_count > 0 else 0.0
                s_neg = (torch.sum(W_current_target[mask_neg]) / (neg_count + 1e-8)).item() if neg_count > 0 else 0.0

                layer_contribution = (mask_pos.to(torch.bfloat16) * s_pos + mask_neg.to(torch.bfloat16) * s_neg)
                W_reconstructed_chunk += layer_contribution
                W_current_target = (W_chunk_gpu - W_reconstructed_chunk) * nonzero_mask.to(torch.bfloat16)

                scalar_layers_accum[pass_idx, 0] += s_pos * (r_end - r_start) / out_features
                scalar_layers_accum[pass_idx, 1] += s_neg * (r_end - r_start) / out_features

                del mask_pos, mask_neg, layer_contribution

            W_final_chunk = W_reconstructed_chunk * nonzero_mask.to(torch.bfloat16)

            Y_pred_chunk = torch.matmul(x_test, W_final_chunk.T)
            ss_res_total += torch.sum((Y_true_chunk.float() - Y_pred_chunk.float()) ** 2).item()
            ss_tot_total += torch.sum((Y_true_chunk.float() - torch.mean(Y_true_chunk.float())) ** 2).item()

            # ЧЕСТНАЯ УПАКОВКА МАСКИ (8 элементов маски -> 1 байт)
            packed_mask = np.packbits(nonzero_mask.cpu().numpy().astype(np.uint8))
            packed_masks_list.append(packed_mask)

            del W_chunk_gpu, W_current_target, W_reconstructed_chunk, W_final_chunk, nonzero_mask, Y_true_chunk, Y_pred_chunk
            torch.cuda.empty_cache()

        matrix_r2 = 1.0 - (ss_res_total / (ss_tot_total + 1e-8))
        full_packed_mask = np.concatenate(packed_masks_list)

        print(f"      [✓] Завершено: R^2 = {matrix_r2:.6f}")

        # Сохранение результатов для каждой сжатой матрицы
        shard_results[f"{tensor_key}.scalars"] = torch.tensor(scalar_layers_accum, dtype=torch.float32)
        shard_results[f"{tensor_key}.thresh"] = torch.tensor([base_thresh_val])
        shard_results[f"{tensor_key}.packed_mask"] = torch.from_numpy(full_packed_mask)

        del x_test, packed_masks_list, full_packed_mask
        gc.collect()

    # Сохранение итога текущего шарда
    output_file = f"deepseek_v4_complete_shard_{shard_idx:05d}.safetensors"
    save_file(shard_results, output_file)

    saved_mb = os.path.getsize(output_file) / (1024**2)
    print(f"\n[+] ИТОГ ШАРДА {shard_idx}: ВСЕ тензоры выкачаны, сжаты и сохранены в '{output_file}' ({saved_mb:.2f} MB)")

    del shard_results
    gc.collect()
