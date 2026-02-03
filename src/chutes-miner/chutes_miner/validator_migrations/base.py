"""
Base class for validator migrations.
"""

from abc import ABC, abstractmethod
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from chutes_common.settings import Validator


class ValidatorMigration(ABC):
    """One-time migration that may call the validator API."""

    @abstractmethod
    async def run(self, session: AsyncSession, validators: List[Validator]) -> None:
        """Run the migration. Raise on any error (pod will exit)."""
        ...
