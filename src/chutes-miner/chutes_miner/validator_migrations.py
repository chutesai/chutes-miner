"""
Validator migrations: one-time steps at Gepetto startup that call the validator
API to sync local state (e.g. re-key servers from UID to hotkey+server_name).

Each migration has a unique key YYYYMMDDXX (date + two-digit index). We run in
sorted order by key. Only migrations after the latest completed key are run
(so deleting old rows from the DB never causes earlier migrations to re-run).
On any failure we do not record and re-raise so the pod exits.
"""

import re
import traceback
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, List

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chutes_common.auth import sign_request
from chutes_common.settings import Validator
from chutes_common.schemas.server import Server
from chutes_common.schemas.validator_migration import ValidatorMigrationRecord

from chutes_miner.api.config import settings
from chutes_miner.api.database import get_session


class ValidatorMigration(ABC):
    """One-time migration that may call the validator API."""

    @abstractmethod
    async def run(self, session: AsyncSession, validators: List[Validator]) -> None:
        """Run the migration. Raise on any error (pod will exit)."""
        ...


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
            servers = result.scalars().all()
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


# Key format: YYYYMMDDXX (10 digits). Enforced at import.
_MIGRATION_KEY_RE = re.compile(r"^\d{10}$")

# Registry: unique key -> migration. Run in sorted order by key.
MIGRATIONS: Dict[str, ValidatorMigration] = {
    "2025013101": SyncServerKeysMigration(),
}


def _assert_migration_keys_unique() -> None:
    """Enforce at build/load time that keys are unique and match YYYYMMDDXX."""
    keys = list(MIGRATIONS.keys())
    assert len(keys) == len(set(keys)), "validator migration keys must be unique"
    for k in keys:
        assert _MIGRATION_KEY_RE.match(k), f"validator migration key must be YYYYMMDDXX, got {k!r}"


_assert_migration_keys_unique()


async def _get_latest_completed_key(session: AsyncSession) -> str | None:
    """Return the latest (max) migration_id that has completed, or None if none."""
    result = await session.execute(select(ValidatorMigrationRecord.migration_id))
    ids = [row[0] for row in result.all()]
    return max(ids) if ids else None


async def run_validator_migrations() -> None:
    """
    Run validator migrations in sorted order by key. Only run migrations after
    the latest completed one (so we never re-run earlier migrations even if
    rows were deleted). On success, record the key. On any failure: do not
    record, re-raise so the pod exits and the issue must be resolved.
    """
    latest_completed: str | None = None
    async with get_session() as session:
        latest_completed = await _get_latest_completed_key(session)

    for key in sorted(MIGRATIONS.keys()):
        if latest_completed is not None and key <= latest_completed:
            continue
        migration = MIGRATIONS[key]
        try:
            async with get_session() as session:
                await migration.run(session, settings.validators)
                session.add(
                    ValidatorMigrationRecord(
                        migration_id=key,
                        name=migration.__class__.__name__,
                    )
                )
                await session.commit()
            logger.success(f"Validator migration {key} ({migration.__class__.__name__}) completed")
        except Exception as exc:
            logger.error(
                f"Validator migration {key} ({migration.__class__.__name__}) failed: {exc}\n{traceback.format_exc()}"
            )
            raise
