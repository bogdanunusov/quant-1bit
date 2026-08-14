
import argparse
import json
import sys
from collections import defaultdict

import numpy as np
from safetensors import safe_open


REQUIRED_COMMON = [".packed_bits", ".shape", ".numel", ".W_mean", ".W_std", ".coverage", ".meta"]

EXPECTED_DTYPE = {
    ".packed_bits": "U8",
    ".shape": "I64",
    ".numel": "I64",
    ".W_mean": "F32",
    ".W_std": "F32",
    ".coverage": "F32",
    ".meta": "F32",
    ".row_scales": "F32",
    ".row_scales.packed_bits": "U8",
    ".row_scales.scales": "F32",
    ".row_scales.packed_numel": "I64",
}

EXPECTED_META_LEN = 6


def infer_layer_type(key: str) -> str:
    """Грубая классификация по имени ключа — только для сводной статистики,
    не влияет на pass/fail отдельных проверок."""
    kl = key.lower()
    if "embed" in kl:
        return "embedding"
    if "lm_head" in kl:
        return "lm_head"
    if "norm" in kl:
        return "norm"
    if any(x in kl for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return "attn"
    if any(x in kl for x in ["gate_proj", "up_proj", "down_proj"]):
        return "ffn"
    if "experts" in kl:
        return "moe"
    return "other"


def validate(input_path: str):
    issues = []          # список dict: {"key":..., "severity":..., "message":...}
    stats = defaultdict(int)
    layer_type_counts = defaultdict(int)
    branch_counts = {"quantized_scales": 0, "plain_scales": 0}

    def add_issue(key, severity, message):
        issues.append({"key": key, "severity": severity, "message": message})
        stats[f"issues_{severity}"] += 1

    print(f"[+] Открываю {input_path} (только заголовки, без загрузки весов в память)")
    with safe_open(input_path, framework="pt") as f:
        all_keys = set(f.keys())
        print(f"[+] Всего ключей в файле: {len(all_keys)}")

        if "__arch_config__" not in all_keys:
            add_issue("__arch_config__", "warning", "отсутствует __arch_config__ -- "
                       "не критично для конвертации в GGUF, но конечный инференс-код "
                       "не сможет восстановить ARCH-словарь без него")

        # --- определяю базовые ключи по схеме main() (key + ".packed_bits") ---
        base_keys = set()
        for k in all_keys:
            if k.endswith(".packed_bits") and ".row_scales." not in k:
                base_keys.add(k[: -len(".packed_bits")])
        base_keys = sorted(base_keys)

        if not base_keys:
            # --- Схема не совпала. Проверяю, не подсунут ли файл из save() по ошибке ---
            save_scheme_hits = [k for k in all_keys if k.endswith("_packed") or k.endswith("_scales")]
            print("\n[!] КРИТИЧНО: не найдено ни одного ключа вида '<key>.packed_bits'.")
            if save_scheme_hits:
                print("    Похоже, это файл из функции save() (напр. "
                      "'..._results.safetensors'), а НЕ из блока main() -> "
                      "'СОХРАНЕНИЕ КВАНТОВАННОЙ МОДЕЛИ' ('..._quantized.safetensors').")
                print(f"    Пример ключей, похожих на схему save(): {save_scheme_hits[:5]}")
                print("    Конвертер (gemma4_to_gguf_v2.py) ожидает файл из main(), "
                      "не из save(). Проверьте, какой файл вы передаёте.")
            else:
                print("    Ключи в файле не соответствуют ни одной известной схеме. "
                      f"Пример ключей из файла: {sorted(all_keys)[:10]}")
            print(f"\n{'='*70}\n  ИТОГ: файл НЕ подходит для конвертации, дальнейшая проверка невозможна\n{'='*70}")
            return 1, issues

        print(f"[+] Найдено {len(base_keys)} тензоров-кандидатов (схема main())")

        # --- прохожу по каждому base_key ---
        for i, base_key in enumerate(base_keys):
            if (i + 1) % 200 == 0:
                print(f"    ...проверено {i+1}/{len(base_keys)}")

            lt = infer_layer_type(base_key)
            layer_type_counts[lt] += 1

            # 1. обязательные общие ключи присутствуют
            missing = [suf for suf in REQUIRED_COMMON if f"{base_key}{suf}" not in all_keys]
            if missing:
                add_issue(base_key, "error", f"отсутствуют обязательные ключи: {missing}")
                continue  # без них дальнейшие проверки этого base_key бессмысленны

            # 2. dtype-проверка для общих полей
            for suf in REQUIRED_COMMON:
                full_key = f"{base_key}{suf}"
                slc = f.get_slice(full_key)
                actual_dtype = slc.get_dtype()
                expected = EXPECTED_DTYPE[suf]
                if actual_dtype != expected:
                    add_issue(full_key, "error",
                               f"dtype={actual_dtype}, ожидался {expected} "
                               f"(конвертер молча даст мусор или упадёт на этом поле)")

            # 3. shape/numel согласованность
            try:
                shape_val = f.get_tensor(f"{base_key}.shape").tolist()
                numel_val = int(f.get_tensor(f"{base_key}.numel").item())
                packed_bits_slc = f.get_slice(f"{base_key}.packed_bits")
                packed_bits_shape = packed_bits_slc.get_shape()
                packed_bits_len = packed_bits_shape[0] if packed_bits_shape else 0

                expected_min_bytes = (numel_val + 7) // 8
                if packed_bits_len < expected_min_bytes:
                    add_issue(base_key, "error",
                               f"packed_bits имеет {packed_bits_len} байт, но numel={numel_val} "
                               f"требует минимум {expected_min_bytes} байт -- будет обрезка бит "
                               f"при распаковке, тихая порча данных")

                declared_numel_from_shape = 1
                for d in shape_val:
                    declared_numel_from_shape *= d
                if declared_numel_from_shape != numel_val:
                    add_issue(base_key, "warning",
                               f"numel={numel_val}, но произведение shape={shape_val} "
                               f"даёт {declared_numel_from_shape} -- несовпадение "
                               f"(может быть намеренным паддингом, но стоит перепроверить)")
            except Exception as e:
                add_issue(base_key, "error", f"не удалось проверить shape/numel: {e}")
                continue

            # 4. ветка A vs Б -- row_scales
            has_quant_scales = f"{base_key}.row_scales.packed_bits" in all_keys
            has_plain_scales = f"{base_key}.row_scales" in all_keys

            if has_quant_scales and has_plain_scales:
                add_issue(base_key, "error",
                           "ОБЕ ветки row_scales присутствуют одновременно "
                           "(.row_scales И .row_scales.packed_bits) -- по логике "
                           "main() должна быть ровно одна; конвертер возьмёт "
                           "квантованную ветку и молча проигнорирует .row_scales")

            elif has_quant_scales:
                branch_counts["quantized_scales"] += 1
                for suf in [".row_scales.packed_bits", ".row_scales.scales", ".row_scales.packed_numel"]:
                    full_key = f"{base_key}{suf}"
                    if full_key not in all_keys:
                        add_issue(base_key, "error", f"ветка Б неполна: отсутствует {full_key}")
                        continue
                    actual_dtype = f.get_slice(full_key).get_dtype()
                    expected = EXPECTED_DTYPE[suf]
                    if actual_dtype != expected:
                        add_issue(full_key, "error", f"dtype={actual_dtype}, ожидался {expected}")

                try:
                    sq_numel = int(f.get_tensor(f"{base_key}.row_scales.packed_numel").item())
                    sq_bits_shape = f.get_slice(f"{base_key}.row_scales.packed_bits").get_shape()
                    sq_bits_len = sq_bits_shape[0] if sq_bits_shape else 0
                    expected_min = (sq_numel + 7) // 8
                    if sq_bits_len < expected_min:
                        add_issue(base_key, "error",
                                   f"row_scales.packed_bits имеет {sq_bits_len} байт, "
                                   f"но packed_numel={sq_numel} требует минимум {expected_min}")

                    num_rows = shape_val[0] if len(shape_val) >= 2 else 1
                    expected_sq_numel = num_rows * 2
                    if sq_numel != expected_sq_numel:
                        add_issue(base_key, "warning",
                                   f"row_scales.packed_numel={sq_numel}, ожидалось "
                                   f"num_rows*2={expected_sq_numel} (num_rows из shape={shape_val})")

                    sq_scales_shape = f.get_slice(f"{base_key}.row_scales.scales").get_shape()
                    if len(sq_scales_shape) != 2 or sq_scales_shape[-1] != 2:
                        add_issue(base_key, "warning",
                                   f"row_scales.scales имеет shape={sq_scales_shape}, "
                                   f"ожидалась форма [N, 2]")
                except Exception as e:
                    add_issue(base_key, "error", f"ошибка проверки ветки Б: {e}")

            elif has_plain_scales:
                branch_counts["plain_scales"] += 1
                try:
                    rs_shape = f.get_slice(f"{base_key}.row_scales").get_shape()
                    num_rows_expected = shape_val[0] if len(shape_val) >= 2 else 1
                    if list(rs_shape) != [num_rows_expected, 2]:
                        add_issue(base_key, "warning",
                                   f"row_scales имеет shape={rs_shape}, ожидалось "
                                   f"[{num_rows_expected}, 2] (num_rows из .shape)")
                except Exception as e:
                    add_issue(base_key, "error", f"ошибка проверки ветки А: {e}")
            else:
                add_issue(base_key, "error",
                           "НИ ОДНА ветка row_scales не найдена -- ни .row_scales, "
                           "ни .row_scales.packed_bits. Тензор не сможет быть "
                           "деквантован, конвертер не найдёт scale-данные.")

            # 5. meta длина
            try:
                meta_shape = f.get_slice(f"{base_key}.meta").get_shape()
                if meta_shape != [EXPECTED_META_LEN]:
                    add_issue(base_key, "warning",
                               f".meta имеет форму {meta_shape}, ожидалось [{EXPECTED_META_LEN}] "
                               f"(r2, rel_error, numel, ratio_used, shardstats_offset, scales_r2)")
            except Exception as e:
                add_issue(base_key, "warning", f"не удалось проверить .meta: {e}")

    n_errors = stats.get("issues_error", 0)
    n_warnings = stats.get("issues_warning", 0)

    print(f"\n{'='*70}")
    print(f"  ИТОГ ПРОВЕРКИ: {len(base_keys)} тензоров")
    print(f"{'='*70}")
    print(f"  Ветка А (row_scales обычные, F32):      {branch_counts['plain_scales']}")
    print(f"  Ветка Б (row_scales квантованы 1-bit):  {branch_counts['quantized_scales']}")
    print(f"\n  Разбивка по layer_type:")
    for lt, cnt in sorted(layer_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {lt:<12} {cnt}")
    print(f"\n  Ошибок (error, блокируют конвертацию):   {n_errors}")
    print(f"  Предупреждений (warning, стоит проверить): {n_warnings}")

    if issues:
        print(f"\n  Первые проблемы:")
        for issue in issues[:20]:
            print(f"    [{issue['severity'].upper()}] {issue['key']}: {issue['message']}")
        if len(issues) > 20:
            print(f"    ... и ещё {len(issues) - 20} в отчёте JSON")

    report_path = input_path + ".validation_report.json"
    with open(report_path, "w") as jf:
        json.dump({
            "input_file": input_path,
            "total_tensors": len(base_keys),
            "branch_counts": branch_counts,
            "layer_type_counts": dict(layer_type_counts),
            "n_errors": n_errors,
            "n_warnings": n_warnings,
            "issues": issues,
        }, jf, indent=2, ensure_ascii=False)
    print(f"\n  Полный отчёт: {report_path}")
    print(f"{'='*70}")

    if n_errors > 0:
        print(f"\n  РЕЗУЛЬТАТ: НЕ РЕКОМЕНДУЕТСЯ конвертировать -- {n_errors} блокирующих ошибок.")
        print(f"  Исправьте их в исходном V11-скрипте / пересохраните файл, затем проверьте снова.")
        return 1, issues
    else:
        print(f"\n  РЕЗУЛЬТАТ: файл чист (0 ошибок), можно передавать в gemma4_to_gguf_v2.py.")
        if n_warnings > 0:
            print(f"  ({n_warnings} предупреждений — не блокируют, но стоит просмотреть отчёт)")
        return 0, issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Проверяет *_quantized.safetensors (схема main()) перед конвертацией в GGUF"
    )
    parser.add_argument("input", help="Путь к *_quantized.safetensors")
    args = parser.parse_args()

    code, _ = validate(args.input)
    sys.exit(code)
