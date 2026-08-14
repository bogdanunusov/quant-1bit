import argparse
import sys

import numpy as np
import torch
from safetensors import safe_open

try:
    import gguf
except ImportError:
    print("ERROR: pip install gguf --break-system-packages", file=sys.stderr)
    sys.exit(1)


MAGIC_KV_PREFIX = "gemma4_1bit"


def load_tensor_keys(f, base_key: str):
    packed_bits = f.get_tensor(f"{base_key}.packed_bits")
    shape = f.get_tensor(f"{base_key}.shape").tolist()
    numel = int(f.get_tensor(f"{base_key}.numel").item())

    all_keys = f.keys()
    scalesq_key = f"{base_key}.row_scales.packed_bits"
    if scalesq_key in all_keys:
        # row_scales САМИ были 1-bit квантованы вашим pack_scales() —
        # разворачиваем их обратно в float32 здесь, на Python-стороне,
        # ДО записи в GGUF. Обоснование то же, что в первой версии
        # конвертера: экономия от двойного квантования row_scales
        # исчезающе мала относительно packed_bits весов, а協 сохранение
        # её как отдельного вложенного формата удвоило бы сложность
        # C++ dequant-кода без реальной выгоды.
        sq_bits = f.get_tensor(f"{base_key}.row_scales.packed_bits")
        sq_scales = f.get_tensor(f"{base_key}.row_scales.scales")
        sq_numel = int(f.get_tensor(f"{base_key}.row_scales.packed_numel").item())
        num_rows = shape[0] if len(shape) >= 2 else 1
        row_scales_shape = (num_rows, 2)

        u = np.unpackbits(sq_bits.cpu().numpy())[:sq_numel]
        signs = torch.from_numpy(u).float()
        signs_2d = signs.reshape(num_rows, -1)[:, :2]
        row_scales = (signs_2d * sq_scales[:, 0:1] + (1 - signs_2d) * sq_scales[:, 1:2]).reshape(row_scales_shape).float()
    else:
        row_scales = f.get_tensor(f"{base_key}.row_scales").float()

    return {
        "packed_bits": packed_bits,
        "row_scales": row_scales,
        "shape": shape,
        "numel": numel,
    }


def convert(input_path: str, output_path: str, arch_name: str = "gemma4"):
    print(f"[+] Открываю {input_path}")
    with safe_open(input_path, framework="pt") as f:
        all_keys = list(f.keys())

        base_keys = set()
        for k in all_keys:
            if k.endswith(".packed_bits") and ".row_scales." not in k:
                base_keys.add(k[: -len(".packed_bits")])
        base_keys = sorted(base_keys)
        print(f"[+] Найдено {len(base_keys)} квантованных тензоров")

        if not base_keys:
            print("ERROR: не найдено ни одного *.packed_bits ключа верхнего уровня.", file=sys.stderr)
            sys.exit(1)

        writer = gguf.GGUFWriter(output_path, arch_name)

        writer.add_string(f"{MAGIC_KV_PREFIX}.format_version", "2")
        writer.add_string(f"{MAGIC_KV_PREFIX}.encoding",
                           "qbits=U8 flat-packed signs (numpy packbits order); "
                           "qscales=F32 [num_rows,2] row-wise (pos,neg); "
                           "dequant: row-major, row_len from KV, MSB-first bit order")
        writer.add_string(f"{MAGIC_KV_PREFIX}.source_script", "gemma4_31b_v11_final.py")

        n_written = 0
        n_failed = 0

        for base_key in base_keys:
            try:
                fields = load_tensor_keys(f, base_key)
                # gguf.GGUFWriter.add_tensor_info автоопределяет GGML-тип по
                # numpy dtype, и её список НЕ включает uint8 напрямую (только
                # F16/F32/F64/I8/I16/I32/I64 -- см. ValueError в исходнике).
                # int8 -- тот же 1 байт/элемент, те же биты; .view() меняет
                # ТОЛЬКО то, как numpy печатает число, не сами байты в памяти
                # (проверено отдельно). Для C++ стороны, которая работает с
                # сырыми байтами через packbits-эквивалентную логику, а не
                # с числовыми значениями, знаковость не имеет значения.
                packed_bits = fields["packed_bits"].cpu().numpy().astype(np.uint8).view(np.int8)
                row_scales = fields["row_scales"].cpu().numpy().astype(np.float32)
                shape = fields["shape"]
                numel = fields["numel"]

                if len(shape) >= 2:
                    num_rows = shape[0]
                    row_len = 1
                    for d in shape[1:]:
                        row_len *= d
                else:
                    num_rows = 1
                    row_len = shape[0] if shape else numel

                # per-tensor metadata через именованные KV-ключи (u32 -- GGUF
                # поддерживает это как стандартный тип значения, никакого
                # изобретения формата)
                writer.add_uint32(f"{MAGIC_KV_PREFIX}.{base_key}.row_len", row_len)
                writer.add_uint32(f"{MAGIC_KV_PREFIX}.{base_key}.num_rows", num_rows)
                writer.add_uint32(f"{MAGIC_KV_PREFIX}.{base_key}.numel", numel)

                # qbits -- обычный U8 тензор. НИКАКОГО raw_dtype= здесь --
                # это тот путь в add_tensor_info, который просто маппит
                # numpy dtype -> GGML тип напрямую (F16/F32/I8 и т.п. ветка),
                # т.е. официально поддерживаемый, не workaround.
                writer.add_tensor(f"{base_key}.qbits", packed_bits)

                # qscales -- обычный F32 тензор, форма [num_rows, 2]
                writer.add_tensor(f"{base_key}.qscales", row_scales)

                n_written += 1

            except Exception as e:
                print(f"  [ERROR] {base_key}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                n_failed += 1

        print(f"\n[+] Записываю {output_path}...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

        print(f"\n{'='*70}")
        print(f"  ГОТОВО: {n_written} тензоров, {n_failed} ошибок")
        print(f"  Выходной файл: {output_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--arch", default="gemma4")
    args = parser.parse_args()
    convert(args.input, args.output, args.arch)

