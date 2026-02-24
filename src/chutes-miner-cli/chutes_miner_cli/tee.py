"""
TEE VM system-manager CLI: shared app and helpers for cache/status commands.
"""

import sys
from typing import Any, Optional

import aiohttp
import typer

from chutes_miner_cli.util import sign_request

TEE_SYSTEM_MANAGER_PORT = 8080

tee_app = typer.Typer(
    help="Commands for TEE VM management.",
    no_args_is_help=True,
)


def build_tee_base_url(ip: str) -> str:
    """Build system-manager base URL for a TEE VM (http://<ip>:8080)."""
    return f"http://{ip}:{TEE_SYSTEM_MANAGER_PORT}"


async def get_tee_server_ip(
    *,
    ip: Optional[str] = None,
    name: Optional[str] = None,
    hotkey: str,
    miner_api: str,
) -> str:
    """
    Get TEE server IP from optional --ip or by resolving --name via API.
    Raises typer.Exit(1) if neither/both provided or resolution fails.
    """
    if ip is not None and name is not None:
        typer.echo("Error: Provide either --ip or --name, not both.", err=True)
        raise typer.Exit(1)
    if ip:
        return ip.strip()
    if name:
        return await resolve_server_by_name(name, hotkey, miner_api)
    typer.echo(
        "Error: Provide either --ip (server IP) or --name (server name to resolve via API).",
        err=True,
    )
    raise typer.Exit(1)


async def resolve_server_by_name(
    name: str,
    hotkey: str,
    miner_api: str,
) -> str:
    """
    Resolve VM IP by server name via miner API. Raises typer.Exit(1) if not found or not TEE.
    Returns ip_address.
    """
    headers, _ = sign_request(hotkey, purpose="management")
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(
            f"{miner_api.rstrip('/')}/servers/",
            headers=headers,
            timeout=30,
        ) as resp:
            servers = await resp.json()
    server = next((s for s in servers if s.get("name") == name), None)
    if not server:
        typer.echo(f"Server not found: {name}", err=True)
        raise typer.Exit(1)
    if not server.get("is_tee"):
        typer.echo(f"Server {name} is not a TEE node.", err=True)
        raise typer.Exit(1)
    ip_address = server.get("ip_address")
    if not ip_address:
        typer.echo(f"Server {name} has no ip_address.", err=True)
        raise typer.Exit(1)
    return ip_address


def _vm_purpose_for_path(path: str) -> str:
    """Return purpose 'status' or 'cache' based on path (for no-body requests)."""
    if path.startswith("/status") or path.startswith("status"):
        return "status"
    if path.startswith("/cache") or path.startswith("cache"):
        return "cache"
    raise ValueError(f"Unsupported API path: {path}")


async def send_tee_request(
    base_url: str,
    path: str,
    method: str,
    hotkey: str,
    *,
    purpose: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> tuple[int, Any]:
    """
    Perform a signed request to the VM system-manager. For no-body requests uses
    purpose 'status' or 'cache' based on path; for body requests signs with payload hash.
    Returns (status_code, data) where data is parsed JSON or raw text.
    """
    path = path if path.startswith("/") else f"/{path}"
    url = f"{base_url.rstrip('/')}{path}"
    if payload is not None:
        headers, payload_string = sign_request(hotkey, payload=payload, remote=True)
    else:
        p = purpose if purpose is not None else _vm_purpose_for_path(path)
        headers, _ = sign_request(hotkey, purpose=p, remote=True)
        payload_string = None

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector, raise_for_status=False) as session:
        if method.upper() == "GET":
            resp = await session.get(url, headers=headers, params=params, timeout=60)
        elif method.upper() == "DELETE":
            resp = await session.delete(url, headers=headers, params=params, timeout=60)
        elif method.upper() == "POST":
            resp = await session.post(
                url, headers=headers, data=payload_string, params=params, timeout=300
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                data = await resp.json()
            except Exception:
                data = await resp.text()
        else:
            data = await resp.text()
        return resp.status, data


async def send_tee_stream_request(
    base_url: str,
    path: str,
    hotkey: str,
    *,
    params: Optional[dict[str, Any]] = None,
) -> int:
    """
    Perform a signed GET request and stream the response body to stdout.
    Used for streaming endpoints (e.g. /status/services/{id}/logs/stream).
    Returns status_code. Raises typer.Exit(1) on non-2xx.
    """
    path = path if path.startswith("/") else f"/{path}"
    url = f"{base_url.rstrip('/')}{path}"
    p = _vm_purpose_for_path(path)
    headers, _ = sign_request(hotkey, purpose=p, remote=True)

    connector = aiohttp.TCPConnector(ssl=False)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with aiohttp.ClientSession(
        connector=connector, raise_for_status=False, timeout=timeout
    ) as session:
        async with session.get(url, headers=headers, params=params) as resp:
            status = resp.status
            if status >= 400:
                body = await resp.text()
                typer.echo(f"Error {status}: {body}", err=True)
                raise typer.Exit(1)
            async for chunk in resp.content.iter_chunked(8192):
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
    return status
