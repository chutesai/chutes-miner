"""
TEE VM cache commands: cache-download, cache-download-status, cache-overview, cache-delete, cache-cleanup.
"""

import asyncio
import json
from typing import Optional

import typer

from chutes_miner_cli.tee import (
    build_tee_base_url,
    send_tee_request,
    resolve_server_by_name,
)


def register(app: typer.Typer) -> None:
    """Register cache commands on the given Typer app."""

    @app.command("cache-download", help="Start (or force) chute download on the TEE VM")
    def cache_download(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        chute_id: str = typer.Option(..., "--chute-id", help="Chute ID to download"),
        force: bool = typer.Option(False, "--force", help="Re-download if already present"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
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
        chute_id: Optional[str] = typer.Option(None, "--chute-id", help="Optional chute ID to filter"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
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
            print(json.dumps(data, indent=2))

        asyncio.run(_run())

    @app.command("cache-overview", help="List cache contents and sizes")
    def cache_overview(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            status, data = await send_tee_request(
                base_url, "/cache/overview", "GET", hotkey
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            print(json.dumps(data, indent=2))

        asyncio.run(_run())

    @app.command("cache-delete", help="Remove cache for one chute")
    def cache_delete(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        chute_id: str = typer.Option(..., "--chute-id", help="Chute ID to remove from cache"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            status, data = await send_tee_request(
                base_url, f"/cache/{chute_id}", "DELETE", hotkey
            )
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
        max_age_days: int = typer.Option(5, "--max-age-days", help="Remove entries older than this many days"),
        max_size_gb: int = typer.Option(100, "--max-size-gb", help="Target max cache size in GB"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
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
            print(json.dumps(data, indent=2))

        asyncio.run(_run())
