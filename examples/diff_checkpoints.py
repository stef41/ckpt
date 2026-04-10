"""Diff two checkpoints — find structural differences between model versions.

Demonstrates diff(), format_diff_rich(), and format_diff_table().
"""

import sys

from ckpt import diff, format_diff, format_diff_rich, format_diff_table


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python diff_checkpoints.py <checkpoint_a> <checkpoint_b>")
        print("  Compare two .safetensors, .bin, or .pt files")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    print(f"Comparing:\n  A: {path_a}\n  B: {path_b}\n")

    # -- Compute diff -----------------------------------------------------
    result = diff(path_a, path_b)

    print(f"Summary:")
    print(f"  Added:   {result.n_added}")
    print(f"  Removed: {result.n_removed}")
    print(f"  Changed: {result.n_changed}")
    print(f"  Shared (identical): {result.n_identical}")
    print()

    # -- Plain text table -------------------------------------------------
    print("=== Diff Table ===")
    print(format_diff_table(result))

    # -- Plain format for logging -----------------------------------------
    print("\n=== Diff (plain) ===")
    print(format_diff(result))

    # -- Rich format (if rich is installed) --------------------------------
    try:
        import rich  # noqa: F401
        print("\n=== Diff (rich) ===")
        print(format_diff_rich(result))
    except ImportError:
        print("\n(install 'rich' for colored diff output)")


if __name__ == "__main__":
    main()
