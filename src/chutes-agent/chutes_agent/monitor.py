# agent/controller/watcher.py
import asyncio
import logging
import os
from typing import Any, Callable, Optional
from urllib.parse import urlparse
from chutes_agent.k8s import KubernetesResourceType
from chutes_common.monitoring.models import ClusterState, MonitoringState, MonitoringStatus
from chutes_common.k8s import WatchEvent, WatchEventType, serializer
from chutes_common.exceptions import (
    ClusterConflictException,
    ClusterNotFoundException,
    ServerNotFoundException,
)
from chutes_agent.exceptions import InvalidOperationError
from kubernetes_asyncio import client, config, watch
from kubernetes_asyncio.client.exceptions import ApiException
from chutes_agent.client import ControlPlaneClient
from chutes_agent.collector import ResourceCollector
from chutes_agent.config import settings
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)


class ResourceMonitor:
    def __init__(self):
        self.control_plane_client: Optional[ControlPlaneClient] = None
        self.collector = ResourceCollector()
        self.core_v1 = None
        self.apps_v1 = None
        self.batch_v1 = None
        self._status = MonitoringStatus(state=MonitoringState.STOPPED)
        self._watcher_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Restart protection
        self._restart_lock = asyncio.Lock()
        self._restart_task: Optional[asyncio.Task] = None

        # Serializes monitoring lifecycle transitions -- start(), stop(), and each
        # background recovery attempt -- so they can never build up or tear down
        # watcher tasks concurrently. Callers coordinate purely through this lock
        # and the monitoring state; nothing has to cancel the recovery loop.
        self._transition_lock = asyncio.Lock()
        # Background loop that retries connecting to the control plane while in the
        # DEGRADED state (e.g. a persisted URL that is temporarily unreachable).
        self._recovery_task: Optional[asyncio.Task] = None

        # Persistence - using mounted host path for persistence across pod restarts
        self._control_plane_url_file = settings.control_plane_url_file

        self.initialize()

    @property
    def status(self):
        return self._status

    @property
    def state(self) -> MonitoringState:
        return self._status.state

    @state.setter
    def state(self, value: MonitoringState):
        self._status.state = value

    def _ensure_state_directory(self):
        """Ensure the state directory exists"""
        state_dir = os.path.dirname(self._control_plane_url_file)
        try:
            os.makedirs(state_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create state directory {state_dir}: {e}")

    def _persist_control_plane_url(self, url: str):
        """Persist control plane URL to file"""
        try:
            self._ensure_state_directory()
            with open(self._control_plane_url_file, "w") as f:
                f.write(url)
            logger.debug(f"Persisted control plane URL to {self._control_plane_url_file}")
        except Exception as e:
            logger.warning(f"Failed to persist control plane URL: {e}")

    def _load_control_plane_url(self) -> Optional[str]:
        """Load a usable control plane URL from file, or None.

        A missing, empty, or malformed file is treated the same as "no URL": we
        return None and wait for the control plane to (re)initiate monitoring via
        ``/start``. It is not raised as an ERROR, because that would 503 the
        liveness probe and crash-loop the pod over a bad persisted file.
        """
        try:
            if os.path.exists(self._control_plane_url_file):
                with open(self._control_plane_url_file, "r") as f:
                    url = f.read().strip()
                if not url:
                    logger.warning("Persisted control plane URL file is empty; ignoring")
                    return None

                parsed_url = urlparse(url)
                if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                    logger.warning(f"Persisted control plane URL is invalid, ignoring: {url!r}")
                    return None

                logger.debug(f"Loaded control plane URL from {self._control_plane_url_file}")
                return url
        except Exception as e:
            logger.warning(f"Failed to load control plane URL: {e}")
        return None

    def _clear_control_plane_url(self):
        """Clear persisted control plane URL"""
        try:
            if os.path.exists(self._control_plane_url_file):
                os.remove(self._control_plane_url_file)
                logger.debug("Cleared persisted control plane URL")
        except Exception as e:
            logger.warning(f"Failed to clear control plane URL: {e}")

    async def _stop_and_clear_url(self):
        """Stop monitoring tasks and forget the persisted control plane URL.

        This is intentional local de-registration: we are giving up on the
        current URL and must NOT auto-resume against it on the next start. Keep it
        distinct from a plain task stop (shutdown/restart), which deliberately
        preserves the persisted URL so ``auto_start`` can resume after a pod
        restart. That is why clearing is tied to this explicit teardown rather
        than to the STOPPED state itself.

        Uses the private ``_stop_monitoring_tasks`` (not the public one) so it
        never cancels a recovery loop that may be its own caller.
        """
        await self._stop_monitoring_tasks()
        self._clear_control_plane_url()

    async def auto_start(self):
        """Auto-start monitoring if a control plane URL is persisted.

        A failed auto-start must never crash the agent. A miner doing a fresh
        launch can carry over a stale control plane URL (e.g. from the base image)
        onto its storage volume; auto-starting against it fails, but the agent API
        server must stay up so the control plane -- or an operator running
        ``add-node``, which calls ``/start`` -- can (re)establish the connection.
        On a connection failure we move to DEGRADED (not ERROR) and retry in the
        background so a transient outage self-heals without operator intervention.
        """
        url = self._load_control_plane_url()
        if not url:
            logger.info(
                "Did not find control plane URL.  Waiting for monitoring to be initiated by control plane."
            )
            return

        logger.info("Found persisted control plane URL, auto-starting monitoring")
        async with self._transition_lock:
            connected = await self._try_connect(url, register=False)
        if not connected:
            self._ensure_recovery_loop()

    async def _try_connect(self, url: str, *, register: bool) -> bool:
        """Attempt to (re)establish monitoring. Must be called holding the transition lock.

        Returns True when there is nothing left to retry -- monitoring is running,
        or we intentionally gave up -- and False on a recoverable connection
        failure (state left as DEGRADED). Never raises for connection-style
        failures; the agent must remain alive.

        ``register`` selects how we announce ourselves to the control plane and
        preserves the pre-existing split (it is not a behavior change): start()
        passes True to register the cluster fresh (replacing it on conflict),
        while auto_start / the recovery loop pass False to send the current
        resource set (registering only if the cluster is unknown).

        The lock is held by the *caller* rather than acquired here: it is
        non-reentrant, and each caller combines this attempt with a step that has
        to be atomic against other transitions -- the recovery loop's DEGRADED
        check, start()'s URL persist -- so the critical section is wider than this
        method. We assert the invariant instead of relying on convention.
        """
        assert self._transition_lock.locked(), (
            "_try_connect must be called while holding _transition_lock"
        )

        # Return value: True means "settled" (running, or intentionally given up),
        # False means "failed recoverably, keep retrying". Defaults to True and
        # only flips on a recoverable connection failure below.
        settled = True

        # Idempotency guard for the start()/recovery-loop race: if monitoring is
        # already running (e.g. the recovery loop connected first), skip -- don't
        # spin up a second set of watcher tasks.
        if self.state != MonitoringState.RUNNING:
            try:
                self.control_plane_client = ControlPlaneClient(url)
                if register:
                    await self._register_cluster()
                else:
                    await self._send_all_resources()
                await self._start_monitoring_tasks()
            except ServerNotFoundException:
                logger.info("Server does not exist in remote inventory, stopping monitoring.")
                await self._stop_and_clear_url()
            except Exception as e:
                self.state = MonitoringState.DEGRADED
                self.status.error_message = f"Not connected to control plane: {str(e)}"
                logger.error(f"Failed to connect to control plane, will keep retrying:\n{str(e)}")
                settled = False

        return settled

    def _ensure_recovery_loop(self):
        """Ensure the background recovery loop is running (idempotent)."""
        if self._recovery_task and not self._recovery_task.done():
            return
        self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def _recovery_loop(self):
        """Retry connecting to the control plane while in the DEGRADED state.

        Coordination is entirely via the transition lock and the monitoring
        state: the loop only acts while state is DEGRADED (its own baseline).
        The moment an explicit start()/stop() -- or a successful attempt -- moves
        state elsewhere, the loop stops on its own. Nothing needs to cancel it.
        This replaces the previous "retry via crash loop": a bad connection no
        longer kills the pod.
        """
        delay = 5
        max_delay = 300  # 5 minutes max
        while True:
            await asyncio.sleep(delay)
            async with self._transition_lock:
                if self.state != MonitoringState.DEGRADED:
                    logger.info(
                        f"Recovery superseded by state '{self.state.value}'; stopping retries"
                    )
                    return
                url = self._load_control_plane_url()
                if not url:
                    logger.info("No control plane URL to recover; stopping retries")
                    # Forget the URL (clears a malformed/leftover file if present)
                    # so we do not sit DEGRADED against something we can't use.
                    await self._stop_and_clear_url()
                    return
                logger.info("Retrying connection to control plane")
                if await self._try_connect(url, register=False):
                    return
            delay = min(delay * 2, max_delay)

    async def start(self, control_plane_url: str):
        async with self._transition_lock:
            self._persist_control_plane_url(control_plane_url)
            connected = await self._try_connect(control_plane_url, register=True)
        if not connected:
            # Keep the agent alive and recover in the background, but tell the
            # caller (control plane / operator) that we are not connected yet.
            self._ensure_recovery_loop()
            raise InvalidOperationError(
                self.status.error_message or "Failed to connect to control plane"
            )

    async def stop(self):
        async with self._transition_lock:
            # Moving out of DEGRADED/RUNNING here also tells the recovery loop
            # (if any) to stop on its next check; stop_monitoring_tasks() cancels
            # it outright for prompt cleanup.
            await self.stop_monitoring_tasks()

            # Clear persisted URL when explicitly stopped
            # Clean up client
            if not self.control_plane_client:
                raise InvalidOperationError(
                    "Agent has no active control plane client. "
                    "Agent may have lost state and cannot remove itself from cache."
                )

            await self.control_plane_client.remove_cluster()
            await self.control_plane_client.close()

            self._clear_control_plane_url()

            await serializer.close()

    async def stop_monitoring_tasks(self):
        # Cancel the background recovery loop, if any. This is NOT needed for
        # correctness -- the loop self-terminates because it re-checks
        # `state == DEGRADED` before acting, so it never resurrects monitoring
        # after a stop. We cancel anyway for prompt teardown (otherwise it sleeps
        # up to max_delay before noticing, and dangles at process shutdown) and to
        # keep a clean slate: _ensure_recovery_loop()'s idempotency guard would
        # treat a stale sleeping loop as "already running" and skip spawning, so a
        # later degraded start could sit idle until the old backoff elapsed.
        # It never calls this method itself (its give-up path uses the private
        # _stop_monitoring_tasks via _stop_and_clear_url), so there is no
        # self-cancellation to guard against.
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
        self._recovery_task = None

        # Cancel any pending restart
        if self._restart_task and not self._restart_task.done():
            self._restart_task.cancel()
            try:
                await self._restart_task
            except asyncio.CancelledError:
                pass

        await self._stop_monitoring_tasks()

    def _restart(self):
        """Initiate a restart with protection against spam restarts"""
        if self._restart_lock.locked():
            logger.info("Restart already in progress, skipping")
            return

        # Cancel any existing restart task
        if self._restart_task and not self._restart_task.done():
            self._restart_task.cancel()

        # Only restart if we are still in a running state
        if self.state == MonitoringState.RUNNING:
            logger.info("Scheduling restart.")
            self._restart_task = asyncio.create_task(self._async_restart())
        else:
            logger.info(f"Skipping restart, monitoring state {self.state=}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=300,  # 5 minutes max
            exp_base=2,
        ),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logger.level("INFO").no),
        after=after_log(logger, logger.level("INFO").no),
    )
    async def _async_restart(self):
        """Async restart with retry logic"""
        async with self._restart_lock:
            logger.info("Executing restart")

            # Update status to restarting
            self.state = MonitoringState.STARTING
            self.status.error_message = "Restarting due to error"

            try:
                # Perform the restart
                await self._stop_monitoring_tasks()
                await self._send_all_resources()
                await self._start_monitoring_tasks()

                logger.info("Restart completed successfully")

            except asyncio.CancelledError:
                logger.info("Restart was cancelled")
                self.state = MonitoringState.STOPPED
                # Don't retry on cancellation
                raise
            except ApiException as e:
                if e.status == 503:
                    logger.error(f"K8s API unavailble: {e}")
                    self.status.error_message = "K8s API unavailable."
                else:
                    logger.error(f"Unexpected exception from K8s API:\n{e}")
                    self.status.error_message = "Unexpected exception from K8s API."

                self.state = MonitoringState.ERROR
                raise

            except Exception as e:
                logger.error(f"Restart attempt failed: {e}")
                # Set error state but let tenacity handle retries
                self.state = MonitoringState.ERROR
                self.status.error_message = f"Restart failed: {str(e)}"
                # Re-raise to trigger tenacity retry
                raise

    async def _start_monitoring_tasks(self):
        """Background task to start monitoring"""
        try:
            # Update status to running
            self.state = MonitoringState.STARTING
            self.status.error_message = None

            # Initialize and start watching
            self._heartbeat_task = asyncio.create_task(self.send_heartbeat())

            # Start the watching process
            self._watcher_task = asyncio.create_task(self._start_watch_resources())

            # Update status to running
            self.state = MonitoringState.RUNNING
            self.status.error_message = None

            logger.info("Monitoring started successfully")

        except asyncio.CancelledError:
            logger.info("Monitoring task was cancelled")
            self.state = MonitoringState.STOPPED
        except Exception as e:
            logger.error(f"Monitoring task failed: {e}")
            self.state = MonitoringState.ERROR
            self.status.error_message = str(e)
            raise

    async def _stop_monitoring_tasks(self):
        """Stop the current monitoring task"""
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        self._watcher_task = None
        self._heartbeat_task = None
        self._status = MonitoringStatus(state=MonitoringState.STOPPED)
        logger.info("Monitoring stopped")

    def initialize(self):
        """Initialize Kubernetes client and send initial resources"""
        try:
            config.load_incluster_config()

            # Initialize API clients
            self.core_v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.batch_v1 = client.BatchV1Api()
            self.networking_v1 = client.NetworkingV1Api()  # For Ingress
            self.rbac_authorization_v1 = client.RbacAuthorizationV1Api()  # For RBAC
            self.storage_v1 = client.StorageV1Api()  # For StorageClass

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    async def _register_cluster(self):
        # Collect and send initial resources
        initial_resources = await self.collector.collect_all_resources()
        try:
            await self.control_plane_client.register_cluster(initial_resources)
        except ClusterConflictException:
            await self.control_plane_client.remove_cluster()
            await self.control_plane_client.register_cluster(initial_resources)

        logger.info("Registered cluster with control plane.")

    async def _send_all_resources(self):
        # Collect and send initial resources
        initial_resources = await self.collector.collect_all_resources()
        try:
            await self.control_plane_client.set_cluster_resources(initial_resources)
        except ClusterNotFoundException:
            await self.control_plane_client.register_cluster(initial_resources)

        logger.info(f"Sent resources for cluster {settings.cluster_name}")

    async def _start_watch_resources(self):
        """Start watching all resource types"""
        try:
            tasks: list[asyncio.Task] = [asyncio.create_task(self.watch_nodes())]
            for namespace in settings.watch_namespaces:
                namespace_tasks = [
                    asyncio.create_task(self.watch_namespaced_deployments(namespace)),
                    asyncio.create_task(self.watch_namespaced_pods(namespace)),
                    asyncio.create_task(self.watch_namespaced_services(namespace)),
                    asyncio.create_task(self.watch_namespaced_jobs(namespace)),
                ]
                tasks += namespace_tasks

            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            self.state = MonitoringState.STOPPED
        except Exception as err:
            self.state = MonitoringState.ERROR
            self.status.error_message = str(err)
            logger.error(f"Exception encoutering while watching resources: {err}")
            self._restart()

    async def watch_namespaced_deployments(self, namespace: str):
        """Watch deployments for changes"""
        await self._watch_resources(
            "deployments", self.apps_v1.list_namespaced_deployment, namespace=namespace
        )

    async def watch_namespaced_pods(self, namespace: str):
        """Watch pods for changes"""
        await self._watch_resources("pods", self.core_v1.list_namespaced_pod, namespace=namespace)

    async def watch_namespaced_services(self, namespace: str):
        """Watch services for changes"""
        await self._watch_resources(
            "services", self.core_v1.list_namespaced_service, namespace=namespace
        )

    async def watch_namespaced_jobs(self, namespace: str):
        """Watch jobs for changes"""
        await self._watch_resources("jobs", self.batch_v1.list_namespaced_job, namespace=namespace)

    async def watch_nodes(self):
        """Watch services for changes"""
        await self._watch_resources("nodes", self.core_v1.list_node)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def _get_initial_resource_version(self, func, **kwargs):
        """Get initial resource version with retry logic"""
        initial_list = await func(**kwargs, watch=False)
        resource_version = initial_list.metadata.resource_version
        logger.debug(f"Got initial resource version: {resource_version}")
        return resource_version

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _process_watch_stream(self, func, resource_version, **kwargs):
        """Process a single watch stream with retry for transient errors"""
        watch_kwargs = kwargs.copy()
        watch_kwargs.update(
            {
                "watch": True,
                "resource_version": resource_version,
                "timeout_seconds": 300,  # 5 minute timeout
                "limit": 200,
            }
        )

        logger.debug(f"Starting watch stream from resource version {resource_version}")

        async with watch.Watch().stream(func, **watch_kwargs) as stream:
            event_count = 0
            stream_start_time = asyncio.get_event_loop().time()

            async for event in stream:
                try:
                    event_count += 1
                    event = WatchEvent.from_dict(event)

                    # Update resource version from each event
                    if hasattr(event.object, "metadata") and hasattr(
                        event.object.metadata, "resource_version"
                    ):
                        resource_version = event.object.metadata.resource_version

                    await self.handle_resource_event(event)

                    # Log progress periodically
                    if event_count % 10 == 0:
                        elapsed = asyncio.get_event_loop().time() - stream_start_time
                        logger.debug(f"Processed {event_count} events in {elapsed:.1f}s")

                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    # Continue processing other events
                    continue

            # Stream ended normally
            elapsed = asyncio.get_event_loop().time() - stream_start_time
            logger.debug(f"Watch stream ended after {elapsed:.1f}s, {event_count} events")

        return resource_version  # Return updated resource version

    async def _watch_resources(
        self, resource_type: str, func: Callable[..., Any], **kwargs
    ) -> None:
        """Watch resources for changes with tenacity-managed retries"""
        resource_version = None

        while True:
            try:
                logger.debug(f"Watching {resource_type}")

                # Get initial resource version if needed
                if resource_version is None:
                    try:
                        resource_version = await self._get_initial_resource_version(func, **kwargs)
                    except Exception as e:
                        logger.error(
                            f"Failed to get initial {resource_type} list after retries: {e}"
                        )
                        await asyncio.sleep(5)
                        continue

                # Process watch stream
                try:
                    resource_version = await self._process_watch_stream(
                        func, resource_version, **kwargs
                    )
                except ApiException as e:
                    if e.status == 410:  # Gone - resource version too old
                        logger.warning(
                            f"{resource_type} resource version {resource_version} too old, resetting"
                        )
                        resource_version = None
                        await asyncio.sleep(1)
                        continue
                    else:
                        # Let tenacity handle other API exceptions
                        raise
                except asyncio.TimeoutError:
                    # Normal timeout, just restart
                    logger.debug(f"{resource_type} watch timed out, restarting (normal)")
                    continue

            except asyncio.CancelledError:
                logger.info(f"{resource_type} watch cancelled")
                break

            except Exception as e:
                logger.error(f"Unhandled error watching {resource_type}: {e}")
                # If we get here, tenacity has exhausted retries
                logger.error(f"Exhausted retries for {resource_type}, triggering restart")
                self._restart()
                break

    async def handle_resource_event(self, event: WatchEvent):
        """Handle a resource change event"""
        try:
            # Special handling for all deletion events
            if event.type == WatchEventType.DELETED:
                # Check if resource still exists in cluster
                still_exists = await self._check_resource_exists(event)

                if still_exists:
                    # Resource still exists, it just got deletionTimestamp - send as TERMINATING
                    terminating_event = WatchEvent(
                        type=WatchEventType.TERMINATING, object=event.object
                    )
                    await self.control_plane_client.send_resource_update(terminating_event)

                    # Schedule a task to check when resource is actually gone
                    asyncio.create_task(self._monitor_resource_actual_deletion(event))

                else:
                    # Resource is actually gone - send normal DELETED event
                    await self.control_plane_client.send_resource_update(event)
            else:
                # Normal event handling for non-deletion events
                await self.control_plane_client.send_resource_update(event)

            logger.debug(
                f"Sent {event.type} event for {event.obj_type}/{event.obj_name} in {event.obj_namespace or 'cluster-scoped'}"
            )

        except Exception as e:
            logger.error(f"Error handling {event.obj_type} event: {e}")

    async def _check_resource_exists(self, event: WatchEvent) -> bool:
        """Check if a resource still exists in the cluster.

        Uses the Kubernetes read API — a successful response means the resource
        exists; a 404 ApiException means it has been removed.
        """
        try:
            resource_type = KubernetesResourceType.from_string(event.obj_type)
        except ValueError:
            logger.warning(f"Unknown resource type: {event.obj_type}")
            return False

        if resource_type.is_namespaced:
            read_func = self._get_namespaced_read_function(resource_type)
            kwargs = {"name": event.obj_name, "namespace": event.obj_namespace}
        else:
            read_func = self._get_cluster_read_function(resource_type)
            kwargs = {"name": event.obj_name}

        if not read_func:
            logger.warning(f"No read function for resource type: {resource_type.value}")
            return False

        try:
            await read_func(**kwargs)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            logger.error(f"Error checking if {event.obj_type} exists: {e}")
            return True

    def _get_namespaced_read_function(self, resource_type: KubernetesResourceType):
        """Get the appropriate read function for namespaced resources"""
        read_functions = {
            KubernetesResourceType.POD: self.core_v1.read_namespaced_pod,
            KubernetesResourceType.SERVICE: self.core_v1.read_namespaced_service,
            KubernetesResourceType.DEPLOYMENT: self.apps_v1.read_namespaced_deployment,
            KubernetesResourceType.JOB: self.batch_v1.read_namespaced_job,
            KubernetesResourceType.CONFIG_MAP: self.core_v1.read_namespaced_config_map,
            KubernetesResourceType.SECRET: self.core_v1.read_namespaced_secret,
            KubernetesResourceType.PERSISTENT_VOLUME_CLAIM: self.core_v1.read_namespaced_persistent_volume_claim,
            KubernetesResourceType.INGRESS: self.networking_v1.read_namespaced_ingress,
            KubernetesResourceType.STATEFUL_SET: self.apps_v1.read_namespaced_stateful_set,
            KubernetesResourceType.DAEMON_SET: self.apps_v1.read_namespaced_daemon_set,
            KubernetesResourceType.REPLICA_SET: self.apps_v1.read_namespaced_replica_set,
            KubernetesResourceType.CRON_JOB: self.batch_v1.read_namespaced_cron_job,
            KubernetesResourceType.NETWORK_POLICY: self.networking_v1.read_namespaced_network_policy,
            KubernetesResourceType.ROLE: self.rbac_authorization_v1.read_namespaced_role,
            KubernetesResourceType.ROLE_BINDING: self.rbac_authorization_v1.read_namespaced_role_binding,
            KubernetesResourceType.SERVICE_ACCOUNT: self.core_v1.read_namespaced_service_account,
            KubernetesResourceType.NAMESPACE: self.core_v1.read_namespace,
        }
        return read_functions.get(resource_type)

    def _get_cluster_read_function(self, resource_type: KubernetesResourceType):
        """Get the appropriate read function for cluster-scoped resources"""
        read_functions = {
            KubernetesResourceType.NODE: self.core_v1.read_node,
            KubernetesResourceType.PERSISTENT_VOLUME: self.core_v1.read_persistent_volume,
            KubernetesResourceType.CLUSTER_ROLE: self.rbac_authorization_v1.read_cluster_role,
            KubernetesResourceType.CLUSTER_ROLE_BINDING: self.rbac_authorization_v1.read_cluster_role_binding,
            KubernetesResourceType.STORAGE_CLASS: self.storage_v1.read_storage_class,
        }
        return read_functions.get(resource_type)

    async def _monitor_resource_actual_deletion(self, original_event: WatchEvent):
        """Monitor for when a resource is actually deleted from the cluster.

        Waits indefinitely until the resource is confirmed gone. Never sends a
        DELETED event until the resource is actually removed to avoid scheduling
        conflicts on standalone clusters.
        """
        check_interval = 2  # Check every 2 seconds
        warn_interval_checks = 60  # Log warning every ~2 minutes

        resource_id = f"{original_event.obj_type}/{original_event.obj_name}"
        if original_event.obj_namespace:
            resource_id += f" in {original_event.obj_namespace}"

        logger.debug(f"Starting deletion monitoring for {resource_id}")

        check_count = 0
        while True:
            try:
                still_exists = await self._check_resource_exists(original_event)

                if not still_exists:
                    # Resource is actually gone now - send the real DELETED event
                    elapsed = check_count * check_interval
                    deleted_event = WatchEvent(
                        type=WatchEventType.DELETED, object=original_event.object
                    )
                    await self.control_plane_client.send_resource_update(deleted_event)
                    logger.debug(f"Resource {resource_id} actually deleted after {elapsed}s")
                    return

                # Resource still exists, wait and check again
                check_count += 1
                if check_count % warn_interval_checks == 0:
                    elapsed = check_count * check_interval
                    logger.warning(f"Resource {resource_id} still terminating after {elapsed}s")

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error monitoring deletion of {resource_id}: {e}")
                # Continue monitoring despite errors
                await asyncio.sleep(check_interval)

    async def send_heartbeat(self):
        """Send periodic heartbeat to control plane"""
        while True:
            try:
                await self.control_plane_client.send_heartbeat(ClusterState.ACTIVE)
                await asyncio.sleep(settings.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed to send heartbeat: {e}")
                # If a heartbeat failed just restart to ensure resources are synced properly.
                self._restart()
                break
