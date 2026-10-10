"""
TEE maintenance commands: check upgrade policy and enter maintenance mode.

These commands hit the validator API (not the VM system-manager), using
hotkey-based auth with X-Chutes-Hotkey / Signature / Nonce headers.

A server's maintenance status is one of:
  none        -- normal operation.
  pending     -- maintenance requested, but sole-survivor instances (the only active instance of
                 their chute) keep serving until a replacement is hot elsewhere or the pending
                 deadline passes. Do NOT reboot yet.
  maintenance -- no instances remain; safe to reboot into the upgrade.
The validator moves pending servers to maintenance automatically; poll
`chutes-miner tee maintenance-status --name <server>` to see when it is safe to reboot.
"""

import asyncio
import json
from typing import Any, Optional

import aiohttp
import typer
from rich.console import Console
from rich.table import Table
from rich import box

from chutes_miner_cli.constants import HOTKEY_ENVVAR, MINER_API_ENVVAR, VALIDATOR_API_ENVVAR
from chutes_miner_cli.util import sign_request, sort_servers

console = Console()

STATUS_STYLES = {
    "none": "No",
    "pending": "[yellow]Pending[/yellow]",
    "maintenance": "[green]Maintenance[/green]",
}


def _server_maintenance_status(server: dict[str, Any]) -> str:
    """Maintenance status from a validator payload (falls back to the legacy boolean)."""
    status = server.get("maintenance_status")
    if status:
        return status
    return "maintenance" if server.get("in_maintenance") else "none"


def display_sole_survivors(survivors: list[dict[str, Any]], title: str) -> None:
    """Render sole-survivor instances (kept serving until replaced elsewhere)."""
    tbl = Table(title=title, box=box.ROUNDED)
    tbl.add_column("Chute ID", style="cyan")
    tbl.add_column("Instance ID")
    for s in survivors:
        tbl.add_row(s.get("chute_id", "-"), s.get("instance_id", "-"))
    console.print(tbl)


def display_maintenance_policy(data: dict[str, Any]) -> None:
    """Render a MaintenancePolicyResponse with rich tables."""
    window = data.get("active_window")
    if window:
        tbl = Table(title="Active Upgrade Window", box=box.ROUNDED)
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value")
        tbl.add_row("Window ID", window.get("id", "-"))
        tbl.add_row("Target Version", window.get("target_measurement_version", "-"))
        tbl.add_row("Start", window.get("upgrade_window_start", "-"))
        tbl.add_row("End", window.get("upgrade_window_end", "-"))
        max_concurrent = window.get("max_concurrent_per_miner", 1)
        tbl.add_row("Max Concurrent / Miner", str(max_concurrent))
        console.print(tbl)

        current = data.get("current_slots", 0)
        console.print(f"\nMaintenance slots: [bold]{current}[/bold] / {max_concurrent} in use")
    else:
        console.print("[yellow]No active upgrade window.[/yellow]")

    servers = data.get("servers") or []
    if servers:
        tbl = Table(title="Servers", box=box.ROUNDED)
        tbl.add_column("Server ID", style="cyan")
        tbl.add_column("Name")
        tbl.add_column("Version")
        tbl.add_column("Needs Upgrade")
        tbl.add_column("Maintenance")
        for s in servers:
            needs = s.get("needs_upgrade", False)
            needs_str = "[yellow]Yes[/yellow]" if needs else "No"
            status = _server_maintenance_status(s)
            tbl.add_row(
                s.get("server_id", "-"),
                s.get("name") or "-",
                s.get("version") or "-",
                needs_str,
                STATUS_STYLES.get(status, status),
            )
        console.print(tbl)
    else:
        console.print("No servers found.")


def display_preflight_denial(data: dict[str, Any]) -> None:
    """Render a PreflightResult that was not eligible."""
    current = data.get("current_slots", 0)
    limit = data.get("limit", 1)
    console.print(f"Maintenance slots: [bold]{current}[/bold] / {limit} in use")

    reasons = data.get("denial_reasons") or []
    if reasons:
        tbl = Table(title="Denial Reasons", box=box.ROUNDED)
        tbl.add_column("Reason", style="red")
        tbl.add_column("Current Ver.")
        tbl.add_column("Target Ver.")
        tbl.add_column("Window ID")
        tbl.add_column("Slots")
        tbl.add_column("Limit")
        for r in reasons:
            slots_str = str(r["current_slots"]) if r.get("current_slots") is not None else "-"
            limit_str = str(r["limit"]) if r.get("limit") is not None else "-"
            tbl.add_row(
                r.get("reason", "-"),
                r.get("current_version") or "-",
                r.get("target_version") or "-",
                r.get("window_id") or "-",
                slots_str,
                limit_str,
            )
        console.print(tbl)


def display_pending_details(data: dict[str, Any], name: str) -> None:
    """Explain a pending server: what it is waiting on, the deadline, and how to poll."""
    awaiting = data.get("awaiting_replacement") or []
    if awaiting:
        display_sole_survivors(awaiting, "Awaiting Replacement")
    deadline = data.get("pending_deadline")
    console.print(
        "[yellow bold]Do NOT reboot yet.[/yellow bold] Sole-survivor instances keep serving "
        "until a replacement is hot elsewhere; the validator then moves this server into "
        "maintenance automatically."
    )
    if deadline:
        console.print(
            f"Unreplaced survivors are purged and the server enters maintenance by: "
            f"[bold]{deadline}[/bold]"
        )
    console.print(
        f"Check when it is safe to reboot with:\n"
        f"  [cyan]chutes-miner tee maintenance-status --name {name}[/cyan]"
    )


def display_maintenance_confirmed(data: dict[str, Any], name: str) -> None:
    """Render a ConfirmMaintenanceResult."""
    status = data.get("maintenance_status", "maintenance")
    if status == "pending":
        console.print("[yellow bold]Maintenance pending.[/yellow bold]")
    else:
        console.print("[green bold]Maintenance confirmed; safe to reboot.[/green bold]")
    console.print(f"Server ID: {data.get('server_id', '-')}")

    purged = data.get("purged_instance_ids") or []
    if purged:
        console.print(f"Purged instances ({len(purged)}):")
        for iid in purged:
            console.print(f"  {iid}")
    else:
        console.print("No instances were purged.")

    window = data.get("window")
    if window:
        tbl = Table(title="Upgrade Window", box=box.ROUNDED)
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value")
        tbl.add_row("Window ID", window.get("id", "-"))
        tbl.add_row("Target Version", window.get("target_measurement_version", "-"))
        tbl.add_row("Start", window.get("upgrade_window_start", "-"))
        tbl.add_row("End", window.get("upgrade_window_end", "-"))
        tbl.add_row("Max Concurrent / Miner", str(window.get("max_concurrent_per_miner", 1)))
        console.print(tbl)

    if status == "pending":
        console.print()
        display_pending_details(data, name)


def display_server_maintenance_status(data: dict[str, Any], name: str) -> None:
    """Render a MaintenanceStatusResponse for a single server."""
    status = data.get("maintenance_status", "none")
    console.print(f"Server: [cyan]{name}[/cyan] ({data.get('server_id', '-')})")
    console.print(f"Maintenance status: {STATUS_STYLES.get(status, status)}")
    if data.get("reconciled_at"):
        console.print(f"Last checked by validator: {data['reconciled_at']}")
    if status == "maintenance":
        console.print("[green bold]Safe to reboot into the upgrade.[/green bold]")
    elif status == "pending":
        display_pending_details(data, name)
    else:
        console.print("Server is not in maintenance.")


def register(app: typer.Typer) -> None:
    """Register maintenance commands on the given Typer app."""

    @app.command(
        "maintenance-status",
        help=(
            "Show upgrade window, concurrency slots, and server maintenance status, or with "
            "--name, whether a single server is safe to reboot (validator API)"
        ),
    )
    def maintenance_status(
        name: Optional[str] = typer.Option(
            None, "--name", "-n", help="Server name or ID to check (poll this while pending)"
        ),
        raw_json: bool = typer.Option(
            False, "--raw-json", help="Output raw JSON for programmatic use"
        ),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        validator_api: str = typer.Option(
            "https://api.chutes.ai",
            help="Validator API base URL",
            envvar=VALIDATOR_API_ENVVAR,
        ),
    ):
        async def _run():
            headers, _ = sign_request(hotkey, purpose="tee", remote=True)
            base = validator_api.rstrip("/")
            if name:
                url = f"{base}/servers/{name}/maintenance"
            else:
                url = f"{base}/servers/maintenance/policy"
            async with aiohttp.ClientSession(raise_for_status=False) as session:
                async with session.get(url, headers=headers, timeout=30) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        typer.echo(f"Error {resp.status}: {body}", err=True)
                        raise typer.Exit(1)
                    data = await resp.json()

            if not name:
                # --name reads that server's own status; the full listing is sorted for display.
                data["servers"] = sort_servers(data.get("servers"))

            if raw_json:
                print(json.dumps(data, indent=2))
            elif name:
                display_server_maintenance_status(data, name)
            else:
                display_maintenance_policy(data)

        asyncio.run(_run())

    @app.command(
        "start-maintenance",
        help="Preflight check and enter maintenance mode for a TEE server (validator API)",
    )
    def start_maintenance(
        name: str = typer.Option(
            ..., "--name", "-n", help="Server name or ID to put into maintenance"
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmation"),
        raw_json: bool = typer.Option(
            False, "--raw-json", help="Output raw JSON for programmatic use"
        ),
        hotkey: str = typer.Option(
            ..., help="Path to the hotkey file for your miner", envvar=HOTKEY_ENVVAR
        ),
        miner_api: str = typer.Option(
            "http://127.0.0.1:32000",
            help="Miner API base URL",
            envvar=MINER_API_ENVVAR,
        ),
        validator_api: str = typer.Option(
            "https://api.chutes.ai",
            help="Validator API base URL",
            envvar=VALIDATOR_API_ENVVAR,
        ),
    ):
        async def _run():
            headers, _ = sign_request(hotkey, purpose="tee", remote=True)
            base = validator_api.rstrip("/")

            async with aiohttp.ClientSession(raise_for_status=False) as session:
                preflight_url = f"{base}/servers/{name}/maintenance/preflight"
                async with session.get(preflight_url, headers=headers, timeout=30) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        typer.echo(f"Preflight error {resp.status}: {body}", err=True)
                        raise typer.Exit(1)
                    preflight = await resp.json()

            if raw_json and not preflight.get("eligible"):
                print(json.dumps(preflight, indent=2))
                raise typer.Exit(1)

            if not preflight.get("eligible"):
                display_preflight_denial(preflight)
                typer.echo("\nServer is not eligible for maintenance.", err=True)
                raise typer.Exit(1)

            current = preflight.get("current_slots", 0)
            limit = preflight.get("limit", 1)
            console.print(f"Maintenance slots: [bold]{current}[/bold] / {limit} in use")
            survivors = preflight.get("sole_survivors") or []
            if survivors:
                display_sole_survivors(survivors, "Sole-Survivor Instances")
                console.print(
                    f"\n[yellow bold]Warning:[/yellow bold] This will lock server "
                    f"[cyan]'{name}'[/cyan] and purge all running instances except the "
                    f"sole survivors above. The server will be [bold]pending[/bold] until "
                    f"those are replaced elsewhere (or the pending deadline passes), then "
                    f"enter maintenance mode automatically."
                )
            else:
                console.print(
                    f"\n[yellow bold]Warning:[/yellow bold] This will lock the server, purge all "
                    f"running instances on server [cyan]'{name}'[/cyan], and enter maintenance "
                    f"mode."
                )

            if not yes:
                typer.confirm("Proceed?", abort=True)

            # Lock the server via the miner API so gepetto stops scheduling new workloads.
            lock_headers, _ = sign_request(hotkey, purpose="management")
            async with aiohttp.ClientSession(raise_for_status=False) as session:
                lock_url = f"{miner_api.rstrip('/')}/servers/{name}/lock"
                async with session.get(lock_url, headers=lock_headers, timeout=30) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        typer.echo(f"Failed to lock server: {resp.status}: {body}", err=True)
                        raise typer.Exit(1)
                    lock_data = await resp.json()
                    console.print(
                        f"Server [cyan]{lock_data.get('name', name)}[/cyan] locked "
                        f"(locked={lock_data.get('locked')})"
                    )

            # Request maintenance on the validator (purges running instances; 202 if pending).
            headers, _ = sign_request(hotkey, purpose="tee", remote=True)
            async with aiohttp.ClientSession(raise_for_status=False) as session:
                confirm_url = f"{base}/servers/{name}/maintenance"
                async with session.put(confirm_url, headers=headers, timeout=60) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        typer.echo(f"Error {resp.status}: {body}", err=True)
                        raise typer.Exit(1)
                    result = await resp.json()

            if raw_json:
                print(json.dumps(result, indent=2))
                return
            display_maintenance_confirmed(result, name)

            console.print(
                f"\n[yellow bold]Reminder:[/yellow bold] The server is locked. "
                f"After the upgrade reboot, unlock it with:\n"
                f"  [cyan]chutes-miner unlock --name {name}[/cyan]"
            )

        asyncio.run(_run())
