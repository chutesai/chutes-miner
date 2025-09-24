from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from chutes_common.monitoring.models import ClusterState


class ResourceSummary(BaseModel):
    """Summary statistics for a resource type"""
    resource_type: str
    count: int
    sample_data: Optional[Dict[str, Any]] = None


class ClusterOverview(BaseModel):
    """Basic cluster information with resource counts"""
    cluster_name: str
    state: ClusterState
    last_heartbeat: Optional[datetime]
    error_message: Optional[str] = None
    is_healthy: bool
    resource_counts: Dict[str, int]  # {"nodes": 3, "pods": 10, etc.}
    total_resources: int


class ClusterDetail(BaseModel):
    """Detailed cluster information including all resources"""
    cluster_name: str
    state: ClusterState
    last_heartbeat: Optional[datetime]
    error_message: Optional[str] = None
    is_healthy: bool
    resources: Dict[str, List[Dict[str, Any]]]  # {"nodes": [...], "pods": [...], etc.}
    resource_summary: List[ResourceSummary]


class HealthSummary(BaseModel):
    """Aggregated health statistics across all clusters"""
    total_clusters: int
    healthy_clusters: int
    unhealthy_clusters: int
    starting_clusters: int
    error_clusters: int
    offline_clusters: int


class DashboardOverview(BaseModel):
    """System-wide overview for dashboard"""
    health_summary: HealthSummary
    total_resources: Dict[str, int]  # Total count of each resource type across all clusters
    cluster_overviews: List[ClusterOverview]
    last_updated: datetime


class ClusterResourceTypeResponse(BaseModel):
    """Response for specific resource type endpoint"""
    cluster_name: str
    resource_type: str
    resources: List[Dict[str, Any]]
    count: int