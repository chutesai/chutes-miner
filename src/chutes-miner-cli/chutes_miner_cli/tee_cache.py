"""
TEE VM cache commands: cache-download, cache-download-status, cache-overview, cache-delete, cache-cleanup.
"""

import asyncio
import json
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from chutes_miner_cli.constants import HOTKEY_ENVVAR, MINER_API_ENVVAR
from chutes_miner_cli.tee import (
    build_tee_base_url,
    send_tee_request,
    resolve_server_by_name,
)

console = Console()


def _format_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.1f} GB"


def _format_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    try:
        from datetime import datetime

        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def display_cache_overview(data: dict[str, Any]) -> None:
    """Pretty-print cache overview (total + table of chutes)."""
    total = data.get("total_size_bytes", 0)
    chutes = data.get("chutes") or []
    console.print(f"[bold]Total cache size:[/bold] {_format_bytes(total)}")
    if not chutes:
        console.print("No cached chutes.")
        return
    table = Table(title="Cache contents", box=box.ROUNDED)
    table.add_column("Chute ID", style="cyan")
    table.add_column("Repo ID", style="green")
    table.add_column("Revision")
    table.add_column("Size", justify="right")
    table.add_column("Last accessed")
    for c in chutes:
        table.add_row(
            c.get("chute_id", "-"),
            c.get("repo_id", "-"),
            c.get("revision") or "-",
            _format_bytes(c.get("size_bytes")),
            _format_ts(c.get("last_accessed")),
        )
    console.print(table)


def display_cache_download_status(data: dict[str, Any]) -> None:
    """Pretty-print download status table."""
    chutes = data.get("chutes") or []
    if not chutes:
        console.print("No chute status entries.")
        return
    table = Table(title="Download status", box=box.ROUNDED)
    table.add_column("Chute ID", style="cyan")
    table.add_column("Status")
    table.add_column("%", justify="right")
    table.add_column("Repo ID")
    table.add_column("Revision")
    table.add_column("Size", justify="right")
    table.add_column("Error")
    for c in chutes:
        status = c.get("status", "-")
        pc = c.get("percent_complete")
        pc_str = f"{pc:.0f}%" if pc is not None else "-"
        err = c.get("error") or "-"
        if err != "-":
            err = f"[red]{err}[/red]"
        table.add_row(
            c.get("chute_id", "-"),
            status,
            pc_str,
            c.get("repo_id") or "-",
            c.get("revision") or "-",
            _format_bytes(c.get("size_bytes")),
            err,
        )
    console.print(table)


def display_cache_cleanup(data: dict[str, Any]) -> None:
    """Pretty-print cleanup result."""
    status = data.get("status", "-")
    freed = data.get("freed_bytes", 0)
    removed = data.get("removed_chutes") or []
    console.print(f"[bold]Status:[/bold] {status}")
    console.print(f"[bold]Freed:[/bold] {_format_bytes(freed)}")
    if removed:
        console.print("[bold]Removed chutes:[/bold]")
        for chute_id in removed:
            console.print(f"  • {chute_id}")
    else:
        console.print("No chutes removed.")


def register(app: typer.Typer) -> None:
    """Register cache commands on the given Typer app."""

    @app.command("cache-download", help="Start (or force) chute download on the TEE VM")
    def cache_download(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        chute_id: str = typer.Option(..., "--chute-id", help="Chute ID to download"),
        force: bool = typer.Option(False, "--force", help="Re-download if already present"),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            status, data = await send_tee_request(
                base_url,
                "/cache/download",
                "POST",
                hotkey,
                payload={"chute_id": chute_id},
                params={"force": str(force).lower()},
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            print(json.dumps(data, indent=2))

        asyncio.run(_run())

    @app.command("cache-download-status", help="Get download status (all or one chute)")
    def cache_download_status(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        chute_id: Optional[str] = typer.Option(
            None, "--chute-id", help="Optional chute ID to filter"
        ),
        raw_json: bool = typer.Option(
            False, "--raw-json", help="Output raw JSON for programmatic use"
        ),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            params = {} if chute_id is None else {"chute_id": chute_id}
            status, data = await send_tee_request(
                base_url, "/cache/download/status", "GET", hotkey, params=params
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if raw_json or not isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                display_cache_download_status(data)

        asyncio.run(_run())

    @app.command("cache-overview", help="List cache contents and sizes")
    def cache_overview(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        raw_json: bool = typer.Option(
            False, "--raw-json", help="Output raw JSON for programmatic use"
        ),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            status, data = await send_tee_request(base_url, "/cache/overview", "GET", hotkey)
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if raw_json or not isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                display_cache_overview(data)

        asyncio.run(_run())

    @app.command("cache-delete", help="Remove cache for one chute")
    def cache_delete(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        chute_id: str = typer.Option(..., "--chute-id", help="Chute ID to remove from cache"),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            status, data = await send_tee_request(base_url, f"/cache/{chute_id}", "DELETE", hotkey)
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data)

        asyncio.run(_run())

    @app.command("cache-cleanup", help="Cleanup cache by age and max size")
    def cache_cleanup(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        max_age_days: int = typer.Option(
            5, "--max-age-days", help="Remove entries older than this many days"
        ),
        max_size_gb: int = typer.Option(100, "--max-size-gb", help="Target max cache size in GB"),
        raw_json: bool = typer.Option(
            False, "--raw-json", help="Output raw JSON for programmatic use"
        ),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            # POST with optional body; use payload for signing
            payload = {"max_age_days": max_age_days, "max_size_gb": max_size_gb}
            status, data = await send_tee_request(
                base_url, "/cache/cleanup", "POST", hotkey, payload=payload
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if raw_json or not isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                display_cache_cleanup(data)

        asyncio.run(_run())
