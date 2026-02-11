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

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chutes_common.schemas.validator_migration import ValidatorMigrationRecord

from chutes_miner.api.config import settings
from chutes_miner.api.database import get_session

from chutes_miner.validator_migrations.base import ValidatorMigration
from chutes_miner.validator_migrations.migrations import SyncServerKeysMigration

# Registry: unique key YYYYMMDDXX -> migration. Run in sorted order by key.
MIGRATIONS: dict[str, ValidatorMigration] = {
    "2025013101": SyncServerKeysMigration(),
}

# Key format: YYYYMMDDXX (10 digits). Enforced at import.
_MIGRATION_KEY_RE = re.compile(r"^\d{10}$")


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
