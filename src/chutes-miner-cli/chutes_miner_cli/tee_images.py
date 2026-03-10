"""
TEE VM image commands: image-list, image-pull, image-pull-status, image-delete, image-prune.
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


_PULL_STATUS_STYLES = {
    "pending": "yellow",
    "in_progress": "yellow",
    "completed": "green",
    "failed": "red",
}


def _styled_pull_status(status: str) -> str:
    style = _PULL_STATUS_STYLES.get(status)
    return f"[{style}]{status}[/{style}]" if style else status


def display_image_list(data: dict[str, Any]) -> None:
    """Pretty-print image list (ref, digest, size)."""
    images = data.get("images") or []
    if not images:
        console.print("No images.")
        return
    table = Table(title="Containerd images", box=box.ROUNDED)
    table.add_column("Ref", style="cyan")
    table.add_column("Digest")
    table.add_column("Size", justify="right")
    for img in images:
        table.add_row(
            img.get("ref", "-"),
            img.get("digest") or "-",
            _format_bytes(img.get("size_bytes")),
        )
    console.print(table)


def display_image_pull_status(data: dict[str, Any]) -> None:
    """Pretty-print pull status table."""
    pulls = data.get("pulls") or []
    if not pulls:
        console.print("No pull status entries.")
        return
    table = Table(title="Image pull status", box=box.ROUNDED)
    table.add_column("Image Ref", style="cyan")
    table.add_column("Status")
    table.add_column("Error")
    for p in pulls:
        status = p.get("status", "-")
        err = p.get("error") or "-"
        if err != "-":
            err = f"[red]{err}[/red]"
        table.add_row(
            p.get("image_ref", "-"),
            _styled_pull_status(status),
            err,
        )
    console.print(table)


def display_image_prune(data: dict[str, Any]) -> None:
    """Pretty-print prune result."""
    status = data.get("status", "-")
    removed_count = data.get("removed_count", 0)
    freed_bytes = data.get("freed_bytes", 0)
    console.print(f"[bold]Status:[/bold] {status}")
    console.print(f"[bold]Removed:[/bold] {removed_count} image(s)")
    console.print(f"[bold]Freed:[/bold] {_format_bytes(freed_bytes)}")


def register(app: typer.Typer) -> None:
    """Register image commands on the given Typer app."""

    @app.command("image-list", help="List images in containerd (GET /images/)")
    def image_list(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
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
            status, data = await send_tee_request(
                base_url, "/images/", "GET", hotkey
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if raw_json or not isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                display_image_list(data)

        asyncio.run(_run())

    @app.command("image-pull", help="Start image pull (POST /images/pull)")
    def image_pull(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        image: str = typer.Option(
            ..., "--image", help="Image: short form (sglang:tag) or full (registry/org/repo:tag)"
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
            status, data = await send_tee_request(
                base_url,
                "/images/pull",
                "POST",
                hotkey,
                payload={"image": image},
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            print(json.dumps(data, indent=2))

        asyncio.run(_run())

    @app.command("image-pull-status", help="Get pull status (all or one image)")
    def image_pull_status(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        image: Optional[str] = typer.Option(
            None, "--image", help="Optional image to filter (short or full form)"
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
            params = {} if image is None else {"image": image}
            status, data = await send_tee_request(
                base_url, "/images/pull/status", "GET", hotkey, params=params
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if raw_json or not isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                display_image_pull_status(data)

        asyncio.run(_run())

    @app.command("image-delete", help="Remove image by reference or ID (DELETE /images/{image})")
    def image_delete(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
        ),
        image: str = typer.Option(
            ..., "--image", help="Image to remove (short or full form, or ID)"
        ),
        force: bool = typer.Option(
            False, "--force", help="Force delete even if in use"
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
            status, data = await send_tee_request(
                base_url,
                f"/images/{image}",
                "DELETE",
                hotkey,
                params={"force": str(force).lower()},
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                print(data)

        asyncio.run(_run())

    @app.command("image-prune", help="Prune unused images (POST /images/prune)")
    def image_prune(
        ip: Optional[str] = typer.Option(
            None, "--ip", help="TEE server IP (use instead of --name to skip API lookup)"
        ),
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="TEE node (server) name (resolve IP via miner API)"
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
            status, data = await send_tee_request(
                base_url, "/images/prune", "POST", hotkey
            )
            if status >= 400:
                typer.echo(f"Error {status}: {data}", err=True)
                raise typer.Exit(1)
            if raw_json or not isinstance(data, dict):
                print(json.dumps(data, indent=2))
            else:
                display_image_prune(data)

        asyncio.run(_run())
