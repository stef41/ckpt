#!/usr/bin/env python3
"""Generate SVG assets for ckpt README."""

from pathlib import Path

from rich.console import Console
from rich.table import Table

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)


def generate_inspect():
    console = Console(record=True, width=110)
    console.print()
    console.print("[bold white]$ ckpt info model.safetensors[/bold white]")
    console.print()
    console.print("[bold]File:[/bold] model.safetensors")
    console.print("[bold]Format:[/bold] safetensors")
    console.print("[bold]Size:[/bold] 13.5 GB")
    console.print("[bold]Parameters:[/bold] 6,738,415,616 (6.74B)")
    console.print("[bold]Tensors:[/bold] 291")
    console.print()

    table = Table(title="Tensors", show_lines=False)
    table.add_column("Name", style="cyan", max_width=52)
    table.add_column("Shape", justify="right")
    table.add_column("DType", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Size", justify="right")

    rows = [
        ("model.embed_tokens.weight", "32000×4096", "BF16", "131,072,000", "262.1 MB"),
        ("model.layers.0.self_attn.q_proj.weight", "4096×4096", "BF16", "16,777,216", "33.6 MB"),
        ("model.layers.0.self_attn.k_proj.weight", "1024×4096", "BF16", "4,194,304", "8.4 MB"),
        ("model.layers.0.self_attn.v_proj.weight", "1024×4096", "BF16", "4,194,304", "8.4 MB"),
        ("model.layers.0.self_attn.o_proj.weight", "4096×4096", "BF16", "16,777,216", "33.6 MB"),
        ("model.layers.0.mlp.gate_proj.weight", "11008×4096", "BF16", "45,088,768", "90.2 MB"),
        ("model.layers.0.mlp.up_proj.weight", "11008×4096", "BF16", "45,088,768", "90.2 MB"),
        ("model.layers.0.mlp.down_proj.weight", "4096×11008", "BF16", "45,088,768", "90.2 MB"),
        ("model.layers.0.input_layernorm.weight", "4096", "BF16", "4,096", "8.2 KB"),
        ("...", "(281 more)", "", "", ""),
        ("model.norm.weight", "4096", "BF16", "4,096", "8.2 KB"),
        ("lm_head.weight", "32000×4096", "BF16", "131,072,000", "262.1 MB"),
    ]
    for row in rows:
        table.add_row(*row)

    console.print(table)
    console.print()

    svg = console.export_svg(title="ckpt inspect — Model Checkpoint Inspector")
    (ASSETS / "inspect.svg").write_text(svg)
    print(f"  ✓ inspect.svg ({len(svg)} bytes)")


def generate_diff():
    console = Console(record=True, width=105)
    console.print()
    console.print("[bold white]$ ckpt diff base_model.safetensors finetuned_model.safetensors[/bold white]")
    console.print()
    console.print("[bold]A:[/bold] base_model.safetensors")
    console.print("[bold]B:[/bold] finetuned_model.safetensors")
    console.print("Shared: 291  Identical: 286  Changes: 7")
    console.print()

    table = Table(show_lines=False)
    table.add_column("", width=3)
    table.add_column("Tensor", style="cyan")
    table.add_column("Change")
    table.add_column("Details")

    table.add_row("[yellow]~[/yellow]", "model.layers.28.self_attn.q_proj.weight", "shape_changed", "4096×4096 → 4096×4352")
    table.add_row("[yellow]~[/yellow]", "model.layers.28.self_attn.k_proj.weight", "shape_changed", "1024×4096 → 1024×4352")
    table.add_row("[yellow]~[/yellow]", "model.layers.28.mlp.gate_proj.weight", "dtype_changed", "BF16 → F16")
    table.add_row("[green]+[/green]", "model.layers.32.self_attn.q_proj.weight", "added", "4096×4096 BF16")
    table.add_row("[green]+[/green]", "model.layers.32.mlp.gate_proj.weight", "added", "11008×4096 BF16")
    table.add_row("[red]-[/red]", "training_metadata.step_count", "removed", "1 F32")
    table.add_row("[red]-[/red]", "training_metadata.loss", "removed", "1 F32")

    console.print(table)
    console.print()

    svg = console.export_svg(title="ckpt diff — Checkpoint Comparison")
    (ASSETS / "diff.svg").write_text(svg)
    print(f"  ✓ diff.svg ({len(svg)} bytes)")


if __name__ == "__main__":
    print("Generating ckpt assets...")
    generate_inspect()
    generate_diff()
    print("Done!")
