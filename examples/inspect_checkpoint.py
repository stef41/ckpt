"""Inspect a model checkpoint — detect format, list tensors, gather stats.

Demonstrates inspect(), detect_format(), format_size(), format_params(),
and stats_from_info().
"""

import sys

from ckpt import detect_format, format_params, format_size, inspect, stats_from_info


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path-to-checkpoint>")
        print("  Supports .safetensors, .bin, .pt, .pth files")
        sys.exit(1)

    path = sys.argv[1]

    # -- Detect format ----------------------------------------------------
    fmt = detect_format(path)
    print(f"File:   {path}")
    print(f"Format: {fmt.value}")
    print()

    # -- Inspect tensors --------------------------------------------------
    info = inspect(path)
    print(f"File size:  {format_size(info.file_size)}")
    print(f"Parameters: {format_params(info.n_parameters)}")
    print(f"Tensors:    {len(info.tensors)}")
    print()

    # Show first 10 tensors
    print(f"{'Name':<50} {'Shape':<20} {'DType':<8} {'Size':>10}")
    print("-" * 90)
    for t in info.tensors[:10]:
        print(f"{t.name:<50} {t.shape_str:<20} {t.dtype.value:<8} {format_size(t.size_bytes):>10}")
    if len(info.tensors) > 10:
        print(f"  ... and {len(info.tensors) - 10} more tensors")
    print()

    # -- Structural stats -------------------------------------------------
    stats = stats_from_info(info)
    print(f"=== Checkpoint Stats ===")
    print(f"  Total size: {stats.total_size_human}")
    print(f"  Dtype counts:")
    for dtype, count in sorted(stats.dtype_counts.items()):
        print(f"    {dtype}: {count} tensors")
    print(f"  Layer groups:")
    for group, n_params in sorted(stats.layer_groups.items(), key=lambda x: -x[1])[:5]:
        print(f"    {group}: {format_params(n_params)}")


if __name__ == "__main__":
    main()
