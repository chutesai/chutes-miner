# app/health/checker.py
import asyncio
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select

from chutes_common.exceptions import AgentError, ClusterConflictException, ClusterNotFoundException
from chutes_common.monitoring.client import start_server_monitoring
from chutes_common.monitoring.models import ClusterState, ClusterStatus
from chutes_common.redis import MonitoringRedisClient
from chutes_common.schemas.server import Server
from chutes_common.k8s import ClusterResources

from chutes_monitor.database import get_session
from chutes_monitor.settings import settings


class HealthChecker:
    """Background service to monitor cluster health based on heartbeat timestamps"""

    _instance: Optional["HealthChecker"] = None

    def __init__(self):
        self.heartbeat_interval = settings.heartbeat_interval
        self.redis_client = MonitoringRedisClient()
        self._running = False
        self._task: asyncio.Task = None

    def __new__(cls, *args, **kwargs):
        """Singleton: return the single HealthChecker instance."""
        if cls._instance is None:
            cls._instance = super().__new__(HealthChecker)
        return cls._instance

    def start(self):
        """Start the health checking service"""
        if self._running:
            logger.warning("Health checker is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Health checker started with {self.heartbeat_interval}s interval, heartbeat timeout: {self.heartbeat_interval * 2}s"
        )

    async def stop(self):
        """Stop the health checking service"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _monitor_loop(self):
        """Main monitoring loop: check cluster health and update status only."""
        while self._running:
            try:
                await self._check_all_clusters()
                await self._cleanup_stale_clusters()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _check_all_clusters(self):
        """Check health of all registered clusters"""
        try:
            cluster_statuses = await self.redis_client.get_all_cluster_statuses()

            for cluster_status in cluster_statuses:
                try:
                    await self._check_cluster_health(cluster_status)
                except Exception as e:
                    logger.error(
                        f"Error checking health for cluster {cluster_status.cluster_name}: {e}"
                    )

        except Exception as e:
            logger.error(f"Error getting cluster statuses for health check: {e}")

    async def _check_cluster_health(self, cluster_status: ClusterStatus):
        """Check health of a specific cluster based on heartbeat timestamp"""
        last_heartbeat = cluster_status.last_heartbeat
        cluster_name = cluster_status.cluster_name

        try:
            healthy = True
            reason = ""
            if not last_heartbeat:
                logger.debug(f"No heartbeat timestamp found for cluster {cluster_name}")
                healthy = False
                reason = "No heartbeat timestamp"
            else:
                # Parse the timestamp
                try:
                    if isinstance(last_heartbeat, str):
                        last_heartbeat_dt = datetime.fromisoformat(
                            last_heartbeat.replace("Z", "+00:00")
                        )
                    else:
                        last_heartbeat_dt = last_heartbeat

                    # Ensure timezone awareness
                    if last_heartbeat_dt.tzinfo is None:
                        last_heartbeat_dt = last_heartbeat_dt.replace(tzinfo=timezone.utc)

                except (ValueError, AttributeError):
                    logger.warning(
                        f"Invalid heartbeat timestamp for cluster {cluster_name}: {last_heartbeat}"
                    )
                    healthy = False
                    reason = "Unable to determine last heartbeat time"

                # Check if heartbeat is stale
                now = datetime.now(timezone.utc)
                heartbeat_timeout = timedelta(seconds=self.heartbeat_interval * 2)

                if now - last_heartbeat_dt > heartbeat_timeout:
                    healthy = False
                    reason = f"Cluster {cluster_name} heartbeat is stale: {last_heartbeat_dt} (timeout: {heartbeat_timeout})"
                    logger.debug(
                        f"Cluster {cluster_name} heartbeat is stale: {last_heartbeat_dt} (timeout: {heartbeat_timeout})"
                    )

            if not healthy:
                await self._mark_cluster_unhealthy(cluster_status, reason)

        except Exception as e:
            logger.error(f"Health check failed for cluster {cluster_name}: {e}")
            await self._mark_cluster_unhealthy(cluster_status, f"Exception encountered: {e}")

    async def _mark_cluster_healthy(self, cluster_name: str):
        """Mark cluster as healthy"""
        try:
            current_status = self.redis_client.get_cluster_status(cluster_name)
            current_state = current_status.get("state") if current_status else None

            # Only update if state has changed to avoid unnecessary Redis writes
            if current_state != ClusterState.ACTIVE:
                timestamp = datetime.now(timezone.utc).isoformat()
                await self.redis_client.update_cluster_status(
                    cluster_name, ClusterState.ACTIVE, timestamp
                )
                logger.info(f"Cluster {cluster_name} marked as healthy")

        except Exception as e:
            logger.error(f"Error marking cluster {cluster_name} as healthy: {e}")

    async def _mark_cluster_unhealthy(self, current_status: ClusterStatus, reason: str):
        """Mark cluster as unhealthy"""
        try:
            cluster_name = current_status.cluster_name
            # Only update if state if currently healthy
            if current_status.is_healthy:
                new_status = ClusterStatus(
                    cluster_name=cluster_name,
                    state=ClusterState.UNHEALTHY,
                    last_heartbeat=current_status.last_heartbeat,
                    error_message=reason,
                )
                await self.redis_client.update_cluster_status(new_status)
                logger.warning(f"Cluster {cluster_name} marked as unhealthy due to stale heartbeat")

        except Exception as e:
            logger.error(f"Error marking cluster {cluster_name} as unhealthy: {e}")

    async def _cleanup_stale_clusters(self):
        """Remove clusters that haven't been seen for a long time"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            cluster_statuses = await self.redis_client.get_all_cluster_statuses()

            for cluster_status in cluster_statuses:
                try:
                    cluster_name = cluster_status.cluster_name

                    if (
                        cluster_status.last_heartbeat < cutoff_time
                        and cluster_status.state != ClusterState.ERROR
                    ):
                        logger.info(f"Cleaning up stale cluster {cluster_name}")
                        # Only clear the resources.  Do not remove health or node info
                        # to avoid triggering a delete event in Gepetto.
                        new_status = ClusterStatus(
                            cluster_name=cluster_name,
                            state=ClusterState.ERROR,
                            last_heartbeat=cluster_status.last_heartbeat,
                            error_message="Cluster offline for more than 1 hour.",
                        )
                        await self.redis_client.update_cluster_status(new_status)
                except Exception as e:
                    # Invalid timestamp format
                    logger.warning(f"Failed to cleanup stale cluster {cluster_name}: {e}")

        except Exception as e:
            logger.error(f"Error during stale cluster cleanup: {e}")


class MonitoringReconciler:
    """
    Reconciliation loop: sync Redis with DB (clear clusters not in DB) and
    reinitiate monitoring for expected-but-unhealthy clusters. Runs on a
    configurable interval separate from the health checker.
    """

    _instance: Optional["MonitoringReconciler"] = None

    def __init__(self):
        self.redis_client = MonitoringRedisClient()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reinitiate_last_attempt: Dict[str, float] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(MonitoringReconciler)
        return cls._instance

    def start(self) -> None:
        """Start the reconciliation loop."""
        if self._running:
            logger.warning("Monitoring reconciler is already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._reconcile())
        logger.info(
            f"Monitoring reconciler started with "
            f"{settings.reconciliation_interval_seconds}s interval"
        )

    async def stop(self) -> None:
        """Stop the reconciliation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring reconciler stopped")

    async def _reconcile(self) -> None:
        """Run cache sync and reinitiate on a timer."""
        while self._running:
            try:
                await asyncio.sleep(settings.reconciliation_interval_seconds)
                if not self._running:
                    break
                await self._reconcile_cache_with_db()
                await self._restore_unhealthy_clusters()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")

    async def _reconcile_cache_with_db(self) -> None:
        """Clear Redis for clusters that are in Redis but whose server is no longer in the DB."""
        try:
            redis_cluster_names = self.redis_client.get_all_cluster_names()
            if not redis_cluster_names:
                return
            async with get_session() as session:
                result = await session.execute(
                    select(Server.name).where(Server.agent_api.isnot(None))
                )
                expected_names: Set[str] = {row[0] for row in result.fetchall()}
            for cluster_name in redis_cluster_names:
                if cluster_name not in expected_names:
                    logger.info(f"Clearing Redis for cluster not in DB: {cluster_name}")
                    try:
                        await self.redis_client.clear_cluster(cluster_name)
                    except Exception as e:
                        logger.warning(f"Failed to clear Redis for {cluster_name}: {e}")
        except Exception as e:
            logger.error(f"Error clearing Redis for clusters not in DB: {e}")

    async def _restore_server_monitoring(self, server: Server) -> None:
        """Reinitiate monitoring for a single agent by sending a signed start-monitoring request. Does not raise."""
        try:
            await start_server_monitoring(
                agent_url=server.agent_api,
                control_plane_url=settings.monitor_api,
                timeout=30,
            )
            logger.info(f"Restored monitoring for {server.name} at {server.agent_api}")
        except AgentError as e:
            logger.warning(
                f"Failed to restore monitoring for {server.name}: {e.status_code} {e.response_text}"
            )
        except Exception as e:
            logger.warning(f"Failed to restore monitoring for {server.name}: {e}")

    async def _restore_unhealthy_clusters(self) -> None:
        """Restore monitoring for expected clusters that are unhealthy or missing in Redis."""
        if not settings.monitor_api:
            return
        throttle_seconds = settings.reinitiate_interval_seconds
        now = time.time()
        try:
            async with get_session() as session:
                result = await session.execute(select(Server).where(Server.agent_api.isnot(None)))
                servers = result.unique().scalars().all()
            cluster_statuses = await self.redis_client.get_all_cluster_statuses()
            status_by_name = {s.cluster_name: s for s in cluster_statuses}
            for server in servers:
                cluster_name = server.name
                status = status_by_name.get(cluster_name)
                needs_reinitiate = (
                    status is None
                    or status.state == ClusterState.UNHEALTHY
                    or status.state == ClusterState.ERROR
                )
                if not needs_reinitiate:
                    continue
                last = self._reinitiate_last_attempt.get(cluster_name, 0)
                if now - last < throttle_seconds:
                    continue
                self._reinitiate_last_attempt[cluster_name] = now
                await self._restore_server_monitoring(server)
        except Exception as e:
            logger.error(f"Error during reinitiate unhealthy clusters: {e}")


class ClusterMonitor:
    """Initiates monitoring workflows on member clusters"""

    _instance: Optional["ClusterMonitor"] = None

    def __init__(self):
        # self.control_plane_url = settings.control_plane_url
        self.redis_client = MonitoringRedisClient()

    def __new__(cls, *args, **kwargs):
        """Singleton: return the single ClusterMonitor instance."""
        if cls._instance is None:
            cls._instance = super().__new__(ClusterMonitor)
        return cls._instance

    async def register_cluster(self, cluster_name: str, resources: ClusterResources) -> bool:
        """Register and start monitoring on a member cluster"""
        try:
            clusters = self.redis_client.get_all_cluster_names()
            if cluster_name in clusters:
                raise ClusterConflictException(f"Cluster {cluster_name} already exists.")

            await self.redis_client.track_cluster(cluster_name, resources)
        except Exception as e:
            logger.error(f"Error registering cluster {cluster_name}: {e}")
            raise

    async def delete_cluster(self, cluster_name: str) -> bool:
        """Completely remove a cluster and its resources"""
        try:
            # Clear all data
            await self.redis_client.clear_cluster(cluster_name)

            logger.info(f"Successfully unregistered cluster {cluster_name}")

        except Exception as e:
            logger.error(f"Error unregistering cluster {cluster_name}: {e}")
            raise

    async def set_cluster_resources(self, cluster_name: str, resources: ClusterResources) -> bool:
        """Set resources for an existing cluster.  If currently being tracked it will overwrite all existing resources."""
        try:
            clusters = self.redis_client.get_all_cluster_names()
            if cluster_name not in clusters:
                raise ClusterNotFoundException(f"Cluster {cluster_name} not found.")

            await self.redis_client.update_cluster_status(
                ClusterStatus(
                    cluster_name=cluster_name,
                    state=ClusterState.STARTING,
                    last_heartbeat=datetime.now(timezone.utc),
                )
            )

            # Clear all data
            await self.redis_client.set_cluster_resources(cluster_name, resources)

            self.redis_client.publish_cluster_reconnect(cluster_name)

            logger.info(f"Successfully set cluster resources for {cluster_name}")

        except Exception as e:
            logger.error(f"Error setting cluster resources {cluster_name}: {e}")
            raise

    async def list_clusters(self) -> List[ClusterStatus]:
        """List all registered clusters with their status"""
        try:
            cluster_names = self.redis_client.get_all_cluster_names()
            clusters = []

            for cluster_name in cluster_names:
                status = self.redis_client.get_cluster_status(cluster_name)
                if status:
                    clusters.append(status)

            return clusters

        except Exception as e:
            logger.error(f"Error listing clusters: {e}")
            return []
