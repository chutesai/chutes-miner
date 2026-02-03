"""
Validator migration tracking ORM.

Records which validator migrations have run so they execute only once.
migration_id is the unique key (YYYYMMDDXX); name is the migration class name.
"""

from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.sql import func
from chutes_common.schemas import Base


class ValidatorMigrationRecord(Base):
    __tablename__ = "validator_migrations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    migration_id = Column(String(10), unique=True, nullable=False)  # YYYYMMDDXX
    name = Column(String, nullable=False)  # migration class name, e.g. SyncServerKeysMigration
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
