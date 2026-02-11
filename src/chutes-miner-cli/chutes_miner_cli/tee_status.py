"""
TEE VM status commands: node-health, services, gpu, disk, shutdown.
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
    get_tee_server_ip,
    send_tee_request,
)

console = Console()


def display_services_list(data: dict[str, Any]) -> None:
    """Pretty-print list of services (id, unit, description)."""
    services = data.get("services") or []
    if not services:
        console.print("No services.")
        return
    table = Table(title="Services", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Unit")
    table.add_column("Description")
    for s in services:
        table.add_row(
            s.get("id", "-"),
            s.get("unit", "-"),
            s.get("description") or "-",
        )
    console.print(table)


def display_overview(data: dict[str, Any]) -> None:
    """Pretty-print system overview (status, services table, gpu summary)."""
    console.print(f"[bold]Status:[/bold] {data.get('status', '-')}")
    console.print(f"[bold]Timestamp:[/bold] {data.get('timestamp', '-')}")
    services = data.get("services") or []
    if services:
        table = Table(title="Services", box=box.ROUNDED)
        table.add_column("Service", style="cyan")
        table.add_column("State")
        table.add_column("Healthy")
        table.add_column("PID")
        for s in services:
            if not isinstance(s, dict):
                table.add_row(str(s), "-", "-", "-")
                continue
            # ServiceStatusResponse: service (ServiceInfo), status (ServiceStatus), healthy, error
            svc = s.get("service") or {}
            name = svc.get("id") or svc.get("unit") or "-"
            healthy = s.get("healthy", False)
            healthy_str = "[green]yes[/green]" if healthy else "[red]no[/red]"
            st = s.get("status")
            if isinstance(st, dict):
                active = st.get("active_state") or "-"
                sub = st.get("sub_state") or "-"
                state_str = f"{active} / {sub}"
                pid = st.get("main_pid") or "-"
            else:
                state_str = "-"
                pid = "-"
            table.add_row(name, state_str, healthy_str, str(pid))
        console.print(table)
    gpu = data.get("gpu")
    if gpu is not None:
        if isinstance(gpu, dict):
            console.print("[bold]GPU:[/bold] (see --raw-json for full nvidia-smi output)")
        else:
            console.print(f"[bold]GPU:[/bold] {gpu}")


def display_disk(data: dict[str, Any]) -> None:
    """Pretty-print disk space (path, total, directories table)."""
    path = data.get("path", "-")
    total = data.get("total_size_human") or data.get("total_size_bytes", "-")
    console.print(f"[bold]Path:[/bold] {path}")
    console.print(f"[bold]Total:[/bold] {total}")
    directories = data.get("directories") or []
    if not directories:
        console.print("No subdirectories.")
        return
    table = Table(title="Directories", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    table.add_column("Size", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("%", justify="right")
    for d in directories:
        pct = d.get("percentage")
        pct_str = f"{pct:.1f}%" if pct is not None else "-"
        table.add_row(
            d.get("name", "-"),
            d.get("path", "-"),
            d.get("size_human", str(d.get("size_bytes", "-"))),
            str(d.get("depth", "-")),
            pct_str,
        )
    console.print(table)


def register(app: typer.Typer) -> None:
    """Register status commands on the given Typer app."""

    @app.command("node-health", help="VM health check (GET /status/health)")
    def node_health(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        async def _run():
            server_ip = await get_tee_server_ip(
                ip=ip, name=name, hotkey=hotkey, miner_api=miner_api
            )
            base_url = build_tee_base_url(server_ip)
            status, data = await send_tee_request(base_url, "/status/health", "GET", hotkey)
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
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        status_service_id: Optional[str] = typer.Option(
            None, "--status", help="Get systemd status for this service ID"
        ),
        logs_service_id: Optional[str] = typer.Option(
            None, "--logs", help="Get journal logs for this service ID"
        ),
        overview: bool = typer.Option(
            False, "--overview", help="Get system overview (services + GPUs)"
        ),
        lines: int = typer.Option(200, "--lines", help="Number of log lines (for --logs)"),
        since_minutes: int = typer.Option(
            0, "--since-minutes", help="Only logs from last N minutes (0 = no filter, for --logs)"
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
        modes = sum([overview, status_service_id is not None, logs_service_id is not None])
        if modes > 1:
            typer.echo(
                "Error: Use exactly one of --overview, --status <id>, or --logs <id>.",
                err=True,
            )
            raise typer.Exit(1)

        async def _run():
            server_ip = await get_tee_server_ip(
                ip=ip, name=name, hotkey=hotkey, miner_api=miner_api
            )
            base_url = build_tee_base_url(server_ip)
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
            status, data = await send_tee_request(base_url, path, "GET", hotkey, params=params)
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if raw_json:
                if isinstance(data, dict):
                    print(json.dumps(data, indent=2))
                else:
                    print(data)
            elif overview and isinstance(data, dict):
                display_overview(data)
            elif not status_service_id and not logs_service_id and isinstance(data, dict):
                display_services_list(data)
            else:
                if isinstance(data, dict):
                    print(json.dumps(data, indent=2))
                else:
                    print(data)

        asyncio.run(_run())

    @app.command("gpu", help="GPU status / nvidia-smi")
    def gpu(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        gpu_index: str = typer.Option("all", "--gpu", help="GPU index or 'all'"),
        detail: bool = typer.Option(False, "--detail", help="Return detailed (-q) output"),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        async def _run():
            server_ip = await get_tee_server_ip(
                ip=ip, name=name, hotkey=hotkey, miner_api=miner_api
            )
            base_url = build_tee_base_url(server_ip)
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
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        path: str = typer.Option("/", "--path", help="Directory path to analyze"),
        diagnostic: bool = typer.Option(False, "--diagnostic", help="Enable diagnostic mode"),
        max_depth: int = typer.Option(
            3, "--max-depth", help="Max depth for diagnostic mode (1-10)"
        ),
        top_n: int = typer.Option(10, "--top-n", help="Show top N directories per level"),
        cross_filesystems: bool = typer.Option(
            False, "--cross-filesystems", help="Cross filesystem boundaries"
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
            server_ip = await get_tee_server_ip(
                ip=ip, name=name, hotkey=hotkey, miner_api=miner_api
            )
            base_url = build_tee_base_url(server_ip)
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
            if raw_json or not isinstance(data, dict):
                if isinstance(data, dict):
                    print(json.dumps(data, indent=2))
                else:
                    print(data)
            else:
                display_disk(data)

        asyncio.run(_run())

    @app.command("shutdown", help="Request system shutdown on the VM (requires --confirm)")
    def shutdown(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        confirm: bool = typer.Option(False, "--confirm", help="Confirm shutdown"),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000", help="Miner API base URL", envvar=MINER_API_ENVVAR
        ),
    ):
        if not confirm:
            typer.echo("Error: --confirm is required to run shutdown.", err=True)
            raise typer.Exit(1)

        async def _run():
            server_ip = await get_tee_server_ip(
                ip=ip, name=name, hotkey=hotkey, miner_api=miner_api
            )
            base_url = build_tee_base_url(server_ip)
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
