"""Request schemas for validator API. Mirrors chutes-api ServerArgs, NodeArgs, MultiNodeArgs."""

from typing import Optional
from pydantic import BaseModel, Field


class NodeArgs(BaseModel):
    """API-compliant GPU payload. Mirrors chutes-api NodeArgs."""

    uuid: str
    name: str
    memory: int
    major: int
    minor: Optional[int] = None
    processors: Optional[int] = None
    sxm: Optional[bool] = None
    clock_rate: float
    max_threads_per_processor: Optional[int] = None
    concurrent_kernels: Optional[bool] = None
    ecc: Optional[bool] = None
    device_index: int = Field(ge=0, lt=10)
    gpu_identifier: str
    verification_host: str
    verification_port: int

    @classmethod
    def from_device_info(
        cls,
        device_info: dict,
        model_short_ref: str,
        device_index: int,
        verification_host: str,
        verification_port: int,
    ) -> "NodeArgs":
        """Parse raw device_info into API-compliant format. Strips extra fields."""
        return cls(
            uuid=device_info["uuid"],
            name=device_info["name"],
            memory=device_info["memory"],
            major=device_info["major"],
            minor=device_info.get("minor"),
            processors=device_info.get("processors"),
            sxm=device_info.get("sxm"),
            clock_rate=device_info["clock_rate"],
            max_threads_per_processor=device_info.get("max_threads_per_processor"),
            concurrent_kernels=device_info.get("concurrent_kernels"),
            ecc=device_info.get("ecc"),
            device_index=device_index,
            gpu_identifier=model_short_ref,
            verification_host=verification_host,
            verification_port=verification_port,
        )


class ServerArgsRequest(BaseModel):
    """API-compliant server registration payload. Mirrors chutes-api ServerArgs (POST /servers/)."""

    host: str
    id: str
    name: Optional[str] = None
    gpus: list[NodeArgs]


class MultiNodeArgsRequest(BaseModel):
    """API-compliant node registration payload. Mirrors chutes-api MultiNodeArgs (POST /nodes/)."""

    server_id: str
    server_name: Optional[str] = None
    nodes: list[NodeArgs]
