"""
TEE VM status commands: node-health, services, gpu, disk, shutdown.
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
    """Register status commands on the given Typer app."""

    @app.command("node-health", help="VM health check (GET /status/health)")
    def node_health(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            status, data = await send_tee_request(
                base_url, "/status/health", "GET", hotkey
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data)

        asyncio.run(_run())

    @app.command(
        "services",
        help="List services, or get status/logs for one, or system overview",
    )
    def services(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        status_service_id: Optional[str] = typer.Option(
            None, "--status", help="Get systemd status for this service ID"
        ),
        logs_service_id: Optional[str] = typer.Option(
            None, "--logs", help="Get journal logs for this service ID"
        ),
        overview: bool = typer.Option(False, "--overview", help="Get system overview (services + GPUs)"),
        lines: int = typer.Option(200, "--lines", help="Number of log lines (for --logs)"),
        since_minutes: int = typer.Option(
            0, "--since-minutes", help="Only logs from last N minutes (0 = no filter, for --logs)"
        ),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
    ):
        modes = sum([overview, status_service_id is not None, logs_service_id is not None])
        if modes > 1:
            typer.echo(
                "Error: Use exactly one of --overview, --status <id>, or --logs <id>.",
                err=True,
            )
            raise typer.Exit(1)

        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            if overview:
                path = "/status/overview"
                params = None
            elif status_service_id:
                path = f"/status/services/{status_service_id}/status"
                params = None
            elif logs_service_id:
                path = f"/status/services/{logs_service_id}/logs"
                params = {"lines": lines, "since_minutes": since_minutes}
            else:
                path = "/status/services"
                params = None
            status, data = await send_tee_request(
                base_url, path, "GET", hotkey, params=params
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data)

        asyncio.run(_run())

    @app.command("gpu", help="GPU status / nvidia-smi")
    def gpu(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        gpu_index: str = typer.Option("all", "--gpu", help="GPU index or 'all'"),
        detail: bool = typer.Option(False, "--detail", help="Return detailed (-q) output"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            status, data = await send_tee_request(
                base_url,
                "/status/gpu/nvidia-smi",
                "GET",
                hotkey,
                params={"gpu": gpu_index, "detail": str(detail).lower()},
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data)

        asyncio.run(_run())

    @app.command("disk", help="Directory sizes (GET /status/disk/space)")
    def disk(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        path: str = typer.Option("/", "--path", help="Directory path to analyze"),
        diagnostic: bool = typer.Option(False, "--diagnostic", help="Enable diagnostic mode"),
        max_depth: int = typer.Option(3, "--max-depth", help="Max depth for diagnostic mode (1-10)"),
        top_n: int = typer.Option(10, "--top-n", help="Show top N directories per level"),
        cross_filesystems: bool = typer.Option(
            False, "--cross-filesystems", help="Cross filesystem boundaries"
        ),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
    ):
        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            params = {
                "path": path,
                "diagnostic": str(diagnostic).lower(),
                "max_depth": max_depth,
                "top_n": top_n,
                "cross_filesystems": str(cross_filesystems).lower(),
            }
            status, data = await send_tee_request(
                base_url, "/status/disk/space", "GET", hotkey, params=params
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data)

        asyncio.run(_run())

    @app.command("shutdown", help="Request system shutdown on the VM (requires --confirm)")
    def shutdown(
        name: str = typer.Option(..., "--name", "-n", help="TEE node (server) name"),
        confirm: bool = typer.Option(False, "--confirm", help="Confirm shutdown"),
        hotkey: str = typer.Option(..., help="Path to the hotkey file for your miner"),
        miner_api: str = typer.Option("http://127.0.0.1:32000", help="Miner API base URL"),
    ):
        if not confirm:
            typer.echo("Error: --confirm is required to run shutdown.", err=True)
            raise typer.Exit(1)

        async def _run():
            ip = await resolve_server_by_name(name, hotkey, miner_api)
            base_url = build_tee_base_url(ip)
            # POST with no body; purpose "status"
            status, data = await send_tee_request(
                base_url, "/status/system/shutdown", "POST", hotkey
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data)

        asyncio.run(_run())
