"""
Validator inventory check for standalone self-registration.

Queries /miner/servers on the validator API to determine if a server
already exists with matching config. Used for idempotent startup.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from chutes_common.auth import sign_request


class InventoryCheckResult(str, Enum):
    """Result of checking validator inventory for a server."""

    MATCH = "match"  # Server exists with matching config, skip registration
    NOT_FOUND = "not_found"  # Server not in inventory, proceed with registration
    CONFLICT = "conflict"  # Server exists but config differs, fail fast


async def check_validator_inventory(
    validator_api: str,
    hotkey: str,
    server_name: str,
    ip_address: str,
    agent_port: int = 32000,
    attestation_port: int = 30443,
) -> InventoryCheckResult:
    """
    Check if a server already exists in the validator inventory with matching config.

    Queries GET {validator_api}/miner/servers (or equivalent) and compares
    name, IP, and ports. Used by standalone scheduler for idempotent startup.

    Args:
        validator_api: Base URL of the validator API (e.g. https://api.chutes.ai)
        hotkey: Miner hotkey (used for auth; response is scoped to this hotkey)
        server_name: Expected server name
        ip_address: Expected server IP
        agent_port: Expected agent port (default 32000)
        attestation_port: Expected attestation port (default 30443)

    Returns:
        InventoryCheckResult.MATCH if server exists with matching config
        InventoryCheckResult.NOT_FOUND if server not in inventory
        InventoryCheckResult.CONFLICT if server exists but config differs
    """
    url = f"{validator_api.rstrip('/')}/miner/servers"
    try:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            headers, _ = sign_request(purpose="miner")
            async with session.get(url, headers=headers) as resp:
                if resp.status == 404:
                    logger.debug(
                        "Validator /miner/servers endpoint not found (404), treating as NOT_FOUND"
                    )
                    return InventoryCheckResult.NOT_FOUND
                servers: List[Dict[str, Any]] = await resp.json()
    except aiohttp.ClientResponseError as exc:
        if exc.status == 404:
            return InventoryCheckResult.NOT_FOUND
        logger.warning(f"Failed to fetch validator inventory: {exc}")
        raise
    except Exception as exc:
        logger.warning(f"Failed to fetch validator inventory: {exc}")
        raise

    matching: Optional[Dict[str, Any]] = None
    for srv in servers:
        if srv.get("name") == server_name:
            matching = srv
            break

    if not matching:
        return InventoryCheckResult.NOT_FOUND

    existing_ip = matching.get("ip_address") or matching.get("host")
    existing_agent_port = matching.get("agent_port", 32000)
    existing_attestation_port = matching.get("attestation_port") or matching.get(
        "verification_port", 30443
    )

    if (
        existing_ip == ip_address
        and existing_agent_port == agent_port
        and existing_attestation_port == attestation_port
    ):
        return InventoryCheckResult.MATCH

    logger.warning(
        f"Server {server_name} exists in validator inventory but config differs: "
        f"expected ip={ip_address} agent_port={agent_port} attestation_port={attestation_port}, "
        f"got ip={existing_ip} agent_port={existing_agent_port} "
        f"attestation_port={existing_attestation_port}"
    )
    return InventoryCheckResult.CONFLICT
