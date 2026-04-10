"""Estimate checkpoint size after dtype conversion or quantization.

Demonstrates estimate_reduction(), estimate_quantized_size(),
and format_estimation().
"""

import sys

from ckpt import (
    estimate_quantized_size,
    estimate_reduction,
    format_estimation,
    format_size,
    inspect,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python estimate_size.py <path-to-checkpoint>")
        sys.exit(1)

    path = sys.argv[1]
    info = inspect(path)
    print(f"Checkpoint: {path}")
    print(f"Original size: {format_size(info.file_size)}")
    print(f"Parameters: {info.n_parameters:,}")
    print()

    # -- Estimate FP32 → FP16 reduction -----------------------------------
    print("=== FP32 → FP16 ===")
    fp16 = estimate_reduction(info, target_dtype="float16")
    print(format_estimation(fp16))
    print()

    # -- Estimate FP32 → BF16 reduction -----------------------------------
    print("=== FP32 → BF16 ===")
    bf16 = estimate_reduction(info, target_dtype="bfloat16")
    print(f"  Estimated: {format_size(bf16.estimated_size)}  "
          f"Reduction: {bf16.reduction_percent:.1f}%")
    print()

    # -- Estimate INT8 reduction ------------------------------------------
    print("=== FP32 → INT8 ===")
    int8 = estimate_reduction(info, target_dtype="int8")
    print(f"  Estimated: {format_size(int8.estimated_size)}  "
          f"Reduction: {int8.reduction_percent:.1f}%")
    print()

    # -- Quantization estimates (4-bit and 8-bit) -------------------------
    print("=== Quantization Estimates ===")
    for bits in (8, 4):
        quant = estimate_quantized_size(info, bits=bits)
        print(f"  {bits}-bit: {format_size(quant.estimated_size)}  "
              f"({quant.reduction_percent:.1f}% smaller)")


if __name__ == "__main__":
    main()
