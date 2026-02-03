from chutes_common.settings import RedisSettings
from pydantic import Field


class MonitorSettings(RedisSettings):
    heartbeat_interval: int = Field(default=30, description="")
    failure_threshold: int = Field(default=1, description="", alias="HEALTH_FAILURE_THRESHOLD")

    sqlalchemy: str = Field(
        default="postgresql+asyncpg://user:password@127.0.0.1:5432/chutes",
        description="",
        alias="POSTGRESQL",
    )

    monitor_api: str = Field(
        default="",
        description="Control plane URL to send to agents when reinitiating monitoring",
        alias="MONITOR_API",
    )

    reinitiate_interval_seconds: int = Field(
        default=120,
        description="Minimum seconds between reinitiate attempts per cluster",
        alias="REINITIATE_INTERVAL_SECONDS",
    )

    reconciliation_interval_seconds: int = Field(
        default=60,
        description="Interval in seconds for the monitoring reconciliation loop",
        alias="RECONCILIATION_INTERVAL_SECONDS",
    )

    debug: bool = Field(default=False, description="")


settings = MonitorSettings()
