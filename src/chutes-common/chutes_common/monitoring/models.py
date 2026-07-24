from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from enum import Enum


class ResourceType(str, Enum):
    ALL = "*"
    NODE = "node"
    DEPLOYMENT = "deployment"
    SERVICE = "service"
    POD = "pod"
    JOB = "job"


class MonitoringState(str, Enum):
    """Monitoring status enumeration"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    # Not the intended state, but recoverable: a control plane URL is known and
    # the agent is retrying the connection in the background. Distinct from ERROR
    # -- the pod stays healthy while it reconnects.
    DEGRADED = "degraded"
    # Terminal, unrecoverable failure. Reserved for cases where restarting the
    # pod is the appropriate escalation (the liveness probe returns 503).
    ERROR = "error"


class MonitoringStatus(BaseModel):
    """Current monitoring status"""

    cluster_id: Optional[str] = None
    state: MonitoringState  # "stopped", "starting", "running", "error"
    error_message: Optional[str] = None
    last_heartbeat: Optional[str] = None


class ClusterState(str, Enum):
    """Cluster monitoring state enumeration"""

    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    ERROR = "error"
    UNHEALTHY = "unhealthy"


class ClusterStatus(BaseModel):
    """Cluster monitoring status"""

    cluster_name: str
    state: ClusterState
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None
    heartbeat_failures: int = 0

    @property
    def is_healthy(self):
        return self.state in [ClusterState.ACTIVE, ClusterState.STARTING]


class HeartbeatData(BaseModel):
    """Heartbeat data from member cluster"""

    state: ClusterState
    timestamp: Optional[str] = None
