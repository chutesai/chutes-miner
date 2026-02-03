"""
Concrete validator migration implementations.

Add new migration classes here; register them in validator_migrations/__init__.py.
"""

import aiohttp
from typing import List

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chutes_common.auth import sign_request
from chutes_common.settings import Validator
from chutes_common.schemas.server import Server

from chutes_miner.api.config import settings
from chutes_miner.validator_migrations.base import ValidatorMigration


class SyncServerKeysMigration(ValidatorMigration):
    """
    Sync server names on the validator: call PATCH /servers/{server_id}
    for each local server so the validator updates vm_name to match our server name.
    """

    async def run(self, session: AsyncSession, validators: List[Validator]) -> None:
        for validator in validators:
            result = await session.execute(
                select(Server).where(Server.validator == validator.hotkey)
            )
            servers = result.unique().scalars().all()
            if not servers:
                continue
            headers, _ = sign_request(purpose="tee")
            base_url = validator.api.rstrip("/")
            async with aiohttp.ClientSession() as client:
                for server in servers:
                    url = f"{base_url}/servers/{server.server_id}"
                    params = {"server_name": server.name}
                    async with client.patch(url, headers=headers, params=params) as resp:
                        if resp.status >= 200 and resp.status < 300:
                            logger.info(
                                f"Validator migration sync_server_keys: patched server {server.server_id} name to {server.name!r} on {validator.hotkey}"
                            )
                        else:
                            text = await resp.text()
                            raise RuntimeError(
                                f"Validator {validator.hotkey} PATCH /servers/{server.server_id} returned {resp.status}: {text}"
                            )
