from datetime import datetime, timezone
from typing import List, Dict, Optional
from chutes_common.k8s import ClusterResources
from chutes_monitor.exceptions import ResourceRetreivalException
from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger

from chutes_common.redis import MonitoringRedisClient
from chutes_common.monitoring.models import ClusterState, ResourceType
from chutes_monitor.api.monitoring.schemas import (
    ClusterDetailResponse,
    ClusterOverview,
    ClusterDetail,
    ClusterResourcesResponse,
    ClusterResourcesResponseItem,
    DashboardOverview,
    HealthSummary,
    ResourceSummary,
)


class MonitoringRouter:
    def __init__(self):
        self.router = APIRouter()
        self._redis_client = None
        self._setup_routes()

    @property
    def redis_client(self):
        """Lazy initialization of redis client"""
        if self._redis_client is None:
            self._redis_client = MonitoringRedisClient()
        return self._redis_client

    def _setup_routes(self):
        """Setup all the monitoring routes"""
        self.router.add_api_route("/clusters", self.list_clusters, methods=["GET"])
        self.router.add_api_route("/clusters/details", self.get_cluster_detail, methods=["GET"])
        self.router.add_api_route(
            "/clusters/resources", self.get_cluster_resources, methods=["GET"]
        )
        # self.router.add_api_route("/clusters_resources", self.get_cluster_resource_type, methods=["GET"])
        self.router.add_api_route("/overview", self.get_dashboard_overview, methods=["GET"])

    async def list_clusters(self) -> List[ClusterOverview]:
        """Get overview of all clusters with basic status and resource counts"""
        try:
            cluster_names = self.redis_client.get_all_cluster_names()
            cluster_overviews = []

            for cluster_name in cluster_names:
                cluster_status = await self.redis_client.get_cluster_status(cluster_name)
                if not cluster_status:
                    continue

                # Get resource counts for this cluster
                try:
                    resource_counts = await self._get_cluster_resource_counts(cluster_name)
                    total_resources = sum(resource_counts.values())
                except ResourceRetreivalException as e:
                    logger.error(f"Error getting resources for cluster {cluster_name}: {e}")
                    resource_counts = {}
                    total_resources = 0

                overview = ClusterOverview(
                    cluster_name=cluster_name,
                    state=cluster_status.state,
                    last_heartbeat=cluster_status.last_heartbeat,
                    error_message=cluster_status.error_message,
                    is_healthy=cluster_status.is_healthy,
                    resource_counts=resource_counts,
                    total_resources=total_resources,
                )
                cluster_overviews.append(overview)

            return cluster_overviews

        except Exception as e:
            logger.error(f"Error listing clusters: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_cluster_detail(
        self,
        cluster_name: Optional[str] = Query(
            None, description="Optional cluster name; omit for all clusters"
        ),
    ) -> ClusterDetailResponse:
        """Get detailed information about a specific cluster including all resources"""
        try:
            cluster_names = self.redis_client.get_all_cluster_names()
            if cluster_name:
                # Check if cluster exists
                if cluster_name not in cluster_names:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Cluster {cluster_name} not found",
                    )
                requested_clusters = [cluster_name]
            else:
                requested_clusters = cluster_names

            details: list[ClusterDetail] = []
            for cluster in requested_clusters:
                details.append(self._get_cluster_details(cluster))

            response = ClusterDetailResponse(clusters=details)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting cluster detail for {cluster_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_cluster_details(self, cluster_name):
        # Get cluster status
        cluster_status = await self.redis_client.get_cluster_status(cluster_name)
        if not cluster_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Cluster status for {cluster_name} not found",
            )

        # Get all resources for this cluster
        resources = await self._get_cluster_resources(cluster_name)

        # Create resource summaries
        resource_summary = []
        for resource_type, resource_list in resources.items():
            summary = ResourceSummary(
                resource_type=resource_type,
                count=len(resource_list),
                sample_data=resource_list[0] if resource_list else None,
            )
            resource_summary.append(summary)

        detail = ClusterDetail(
            cluster_name=cluster_name,
            state=cluster_status.state,
            last_heartbeat=cluster_status.last_heartbeat,
            error_message=cluster_status.error_message,
            is_healthy=cluster_status.is_healthy,
            resources=resources.to_dict(),
            resource_summary=resource_summary,
        )

        return detail

    async def get_cluster_resources(
        self,
        cluster_name: Optional[str] = Query(
            None, description="Optional cluster name; omit for all clusters"
        ),
        resource_type: Optional[str] = Query(
            None, description="Optional resource type to filter for"
        ),
    ) -> ClusterResourcesResponse:
        """Get all resources for a cluster, grouped by type"""
        try:
            cluster_names = self.redis_client.get_all_cluster_names()
            if cluster_name:
                # Check if cluster exists
                if cluster_name not in cluster_names:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Cluster {cluster_name} not found",
                    )
                requested_clusters = [cluster_name]
            else:
                requested_clusters = cluster_names

            if resource_type:
                try:
                    _resource_type = ResourceType(resource_type)
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid resource type supplied: {resource_type}",
                    )
            else:
                _resource_type = ResourceType.ALL

            response_items: list[ClusterResourcesResponseItem] = []
            for cluster in requested_clusters:
                resources = await self._get_cluster_resources(cluster, _resource_type)
                response_items.append(
                    ClusterResourcesResponseItem(
                        cluster_name=cluster, resources=resources.to_dict(), count=resources.count
                    )
                )

            response = ClusterResourcesResponse(clusters=response_items)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting resources for cluster {cluster_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_dashboard_overview(self) -> DashboardOverview:
        """Get comprehensive dashboard overview with system-wide statistics"""
        try:
            cluster_overviews = await self.list_clusters()

            # Calculate health summary
            health_summary = self._calculate_health_summary(cluster_overviews)

            # Calculate total resources across all clusters
            total_resources = self._calculate_total_resources(cluster_overviews)

            overview = DashboardOverview(
                health_summary=health_summary,
                total_resources=total_resources,
                cluster_overviews=cluster_overviews,
                last_updated=datetime.now(timezone.utc),
            )

            return overview

        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_cluster_resource_counts(self, cluster_name: str) -> Dict[str, int]:
        """Get count of resources by type for a cluster"""
        resource_counts = {}

        try:
            resources = self.redis_client.get_resources(cluster_name)
            for resource_type, resources in resources.items():
                resource_counts[resource_type.value] = len(resources)
        except Exception as e:
            logger.error(f"Error getting {resource_type} count for {cluster_name}: {e}")
            raise ResourceRetreivalException(
                f"Failed to retrieve resource counts for {cluster_name}"
            )

        return resource_counts

    async def _get_cluster_resources(
        self, cluster_name: str, resource_type: ResourceType = ResourceType.ALL
    ) -> ClusterResources:
        """Get all resources for a cluster, organized by type"""

        try:
            cluster_resources = self.redis_client.get_resources(cluster_name, resource_type)
        except Exception as e:
            logger.debug(f"Error getting resources for {cluster_name}: {e}")
            raise ResourceRetreivalException(f"Failed to retrieve resources for {cluster_name}")

        return cluster_resources

    def _calculate_health_summary(self, cluster_overviews: List[ClusterOverview]) -> HealthSummary:
        """Calculate health statistics from cluster overviews"""
        total_clusters = len(cluster_overviews)
        healthy_clusters = 0
        unhealthy_clusters = 0
        starting_clusters = 0
        error_clusters = 0
        offline_clusters = 0

        for cluster in cluster_overviews:
            if cluster.state == ClusterState.ACTIVE:
                healthy_clusters += 1
            elif cluster.state == ClusterState.UNHEALTHY:
                unhealthy_clusters += 1
            elif cluster.state == ClusterState.STARTING:
                starting_clusters += 1
            elif cluster.state == ClusterState.ERROR:
                error_clusters += 1
            else:
                offline_clusters += 1

        return HealthSummary(
            total_clusters=total_clusters,
            healthy_clusters=healthy_clusters,
            unhealthy_clusters=unhealthy_clusters,
            starting_clusters=starting_clusters,
            error_clusters=error_clusters,
            offline_clusters=offline_clusters,
        )

    def _calculate_total_resources(
        self, cluster_overviews: List[ClusterOverview]
    ) -> Dict[str, int]:
        """Calculate total resources across all clusters"""
        total_resources = {}

        for cluster in cluster_overviews:
            for resource_type, count in cluster.resource_counts.items():
                total_resources[resource_type] = total_resources.get(resource_type, 0) + count

        return total_resources


# Create the router instance
monitoring_router = MonitoringRouter()
router = monitoring_router.router
