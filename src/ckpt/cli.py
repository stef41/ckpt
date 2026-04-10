"""CLI for ckpt."""

from __future__ import annotations

import sys

try:
    import click
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]


def _build_cli():  # type: ignore[no-untyped-def]
    if not _HAS_CLICK:
        return None

    from ckpt.diff import diff, format_diff
    from ckpt.inspect import format_params, format_size, inspect
    from ckpt.stats import format_stats, stats_from_info
    from ckpt.validate import validate

    @click.group()
    @click.version_option(package_name="ckpt")
    def cli() -> None:
        """ckpt — inspect, convert, diff, and merge model checkpoints."""

    @cli.command()
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    def info(path: str, as_json: bool) -> None:
        """Inspect a checkpoint file."""
        import json as json_mod
        ckpt_info = inspect(path)

        if as_json:
            data = {
                "path": ckpt_info.path,
                "format": ckpt_info.format.value,
                "file_size": ckpt_info.file_size,
                "n_tensors": ckpt_info.n_tensors,
                "n_parameters": ckpt_info.n_parameters,
                "tensors": [
                    {"name": t.name, "shape": t.shape, "dtype": t.dtype.value}
                    for t in ckpt_info.tensors
                ],
            }
            click.echo(json_mod.dumps(data, indent=2))
            return

        if _HAS_RICH and _console is not None:
            _console.print(f"\n[bold]File:[/bold] {ckpt_info.path}")
            _console.print(f"[bold]Format:[/bold] {ckpt_info.format.value}")
            _console.print(f"[bold]Size:[/bold] {format_size(ckpt_info.file_size)}")
            _console.print(f"[bold]Parameters:[/bold] {ckpt_info.n_parameters:,} ({format_params(ckpt_info.n_parameters)})")
            _console.print(f"[bold]Tensors:[/bold] {ckpt_info.n_tensors}")
            _console.print()

            table = Table(title="Tensors", show_lines=False)
            table.add_column("Name", style="cyan")
            table.add_column("Shape", justify="right")
            table.add_column("DType", justify="center")
            table.add_column("Params", justify="right")
            table.add_column("Size", justify="right")

            for t in ckpt_info.tensors[:50]:
                table.add_row(
                    t.name, t.shape_str, t.dtype.value,
                    f"{t.numel:,}", format_size(t.size_bytes),
                )
            if len(ckpt_info.tensors) > 50:
                table.add_row("...", f"({len(ckpt_info.tensors) - 50} more)", "", "", "")

            _console.print(table)
        else:
            click.echo(f"File: {ckpt_info.path}")
            click.echo(f"Format: {ckpt_info.format.value}")
            click.echo(f"Parameters: {ckpt_info.n_parameters:,}")
            for t in ckpt_info.tensors[:50]:
                click.echo(f"  {t.name}: {t.shape} {t.dtype.value}")

    @cli.command(name="diff")
    @click.argument("path_a", type=click.Path(exists=True))
    @click.argument("path_b", type=click.Path(exists=True))
    def diff_cmd(path_a: str, path_b: str) -> None:
        """Compare two checkpoint files."""
        result = diff(path_a, path_b)

        if _HAS_RICH and _console is not None:
            _console.print(f"\n[bold]A:[/bold] {result.path_a}")
            _console.print(f"[bold]B:[/bold] {result.path_b}")
            _console.print(f"Shared: {result.n_shared}  Identical: {result.n_identical}  Changes: {result.n_changes}")
            _console.print()

            if result.entries:
                table = Table(show_lines=False)
                table.add_column("", style="bold", width=3)
                table.add_column("Tensor", style="cyan")
                table.add_column("Change")
                table.add_column("Details")

                for e in result.entries:
                    symbols = {"added": "[green]+[/green]", "removed": "[red]-[/red]",
                             "shape_changed": "[yellow]~[/yellow]", "dtype_changed": "[yellow]~[/yellow]"}
                    table.add_row(symbols.get(e.change_type, "?"), e.tensor_name, e.change_type, e.details)
                _console.print(table)
            else:
                _console.print("[green]Checkpoints are structurally identical.[/green]")
        else:
            click.echo(format_diff(result))

    @cli.command()
    @click.argument("path", type=click.Path(exists=True))
    def stats(path: str) -> None:
        """Show checkpoint statistics."""
        ckpt_info = inspect(path)
        ckpt_stats = stats_from_info(ckpt_info)
        click.echo(format_stats(ckpt_stats))

    @cli.command(name="validate")
    @click.argument("path", type=click.Path(exists=True))
    def validate_cmd(path: str) -> None:
        """Validate checkpoint integrity."""
        result = validate(path)
        if result.valid:
            click.echo(f"✓ {path}: valid ({result.format.value})")
        else:
            click.echo(f"✗ {path}: invalid")
        for issue in result.issues:
            prefix = "  ERROR:" if issue.severity == "error" else "  WARN:"
            click.echo(f"{prefix} {issue.message}")
        if not result.valid:
            raise SystemExit(1)

    return cli


cli = _build_cli()


def main() -> None:
    if cli is None:
        print(
            "The CLI requires extra dependencies. Install with:\n"
            "  pip install ckpt[cli]",
            file=sys.stderr,
        )
        sys.exit(1)
    cli()


if __name__ == "__main__":
    main()
