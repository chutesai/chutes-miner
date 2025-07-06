from functools import lru_cache
import math
import time
import uuid
import traceback
import abc
from loguru import logger
from typing import Generator, List, Dict, Any, Optional, Tuple, Union
from kubernetes import watch
from kubernetes.client import (
    V1Deployment,
    V1Pod,
    V1Service,
    V1Node,
    V1NodeList,
    V1PodList,
    V1ObjectMeta,
    V1DeploymentList,
    V1ConfigMap,
)
from kubernetes.client.rest import ApiException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from chutes_miner.api.exceptions import DeploymentFailure
from chutes_miner.api.config import k8s_api_client, k8s_custom_objects_client, settings
from chutes_miner.api.database import get_session
from chutes_miner.api.k8s.constants import (
    CHUTE_CODE_CM_PREFIX,
    CHUTE_DEPLOY_PREFIX,
    CHUTE_PP_PREFIX,
    CHUTE_SVC_PREFIX,
    GRAVAL_DEPLOY_PREFIX,
    GRAVAL_PP_PREFIX,
    GRAVAL_SVC_PREFIX,
    SEARCH_DEPLOYMENTS_PATH,
    SEARCH_NODES_PATH,
    SEARCH_PODS_PATH,
)
from chutes_miner.api.k8s.karmada.models import (
    ClusterAffinity,
    WatchEvent,
    Placement,
    PropagationPolicy,
    ReplicaScheduling,
    ResourceSelector,
    WatchEventType,
)
from chutes_miner.api.k8s.response import ApiResponse
from chutes_miner.api.k8s.util import build_chute_deployment, build_chute_service
from chutes_miner.api.server.schemas import Server
from chutes_miner.api.chute.schemas import Chute
from chutes_miner.api.deployment.schemas import Deployment
from chutes_miner.api.config import k8s_core_client, k8s_app_client


# Abstract base class for all Kubernetes operations
class K8sOperator(abc.ABC):
    """Base class for Kubernetes operations that works with both single-cluster and Karmada setups."""

    _instance: Optional["K8sOperator"] = None

    def __new__(cls, *args, **kwargs):
        """
        Factory method that creates either a SingleClusterK8sOperator or KarmadaK8sOperator
        based on the detected infrastructure.
        """
        # If we already have an instance, return it (singleton pattern)
        if cls._instance is not None:
            return cls._instance

        # If someone is trying to instantiate the concrete classes directly, let them
        if cls is not K8sOperator:
            instance = super().__new__(cls)
            return instance

        # Otherwise, determine which implementation to use
        try:
            # Detection logic
            nodes = k8s_core_client().list_node(label_selector="karmada-control-plane=true")
            if nodes.items:
                logger.debug("Creating K8S Operator for Karmada")
                cls._instance = super().__new__(KarmadaK8sOperator)
            else:
                logger.debug("Creating K8S Operator for K3S")
                cls._instance = super().__new__(SingleClusterK8sOperator)
        except Exception:
            cls._instance = super().__new__(SingleClusterK8sOperator)

        return cls._instance

    def _extract_deployment_info(self, deployment: V1Deployment) -> Dict:
        """
        Extract deployment info from the deployment objects.
        """
        deploy_info = {
            "uuid": deployment.metadata.uid,
            "deployment_id": deployment.metadata.labels.get("chutes/deployment-id"),
            "name": deployment.metadata.name,
            "namespace": deployment.metadata.namespace,
            "labels": deployment.metadata.labels,
            "chute_id": deployment.metadata.labels.get("chutes/chute-id"),
            "version": deployment.metadata.labels.get("chutes/version"),
            "node_selector": deployment.spec.template.spec.node_selector,
        }
        deploy_info["ready"] = self._is_deployment_ready(deployment)
        pods = self._get_pods(
            namespace=deployment.metadata.namespace,
            label_selector=deployment.spec.selector.match_labels,
        )
        deploy_info["pods"] = []
        for pod in pods.items:
            state = (
                pod.status.container_statuses[0].state if pod.status.container_statuses else None
            )
            last_state = (
                pod.status.container_statuses[0].last_state
                if pod.status.container_statuses
                else None
            )
            pod_info = {
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "restart_count": pod.status.container_statuses[0].restart_count
                if pod.status.container_statuses
                else 0,
                "state": {
                    "running": state.running.to_dict() if state and state.running else None,
                    "terminated": state.terminated.to_dict()
                    if state and state.terminated
                    else None,
                    "waiting": state.waiting.to_dict() if state and state.waiting else None,
                }
                if state
                else None,
                "last_state": {
                    "running": last_state.running.to_dict()
                    if last_state and last_state.running
                    else None,
                    "terminated": last_state.terminated.to_dict()
                    if last_state and last_state.terminated
                    else None,
                    "waiting": last_state.waiting.to_dict()
                    if last_state and last_state.waiting
                    else None,
                }
                if last_state
                else None,
            }
            deploy_info["pods"].append(pod_info)
            deploy_info["node"] = pod.spec.node_name
        return deploy_info

    @abc.abstractmethod
    def watch_pods(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ) -> Generator[WatchEvent, None, None]:
        """
        Watch pods and yield events with type and pod object.

        Yields:
            dict: Event with 'type' ('ADDED', 'MODIFIED', 'DELETED') and 'object' (V1Pod)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_pods(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ) -> V1PodList:
        """Get all Kubernetes nodes via k8s client, optionally filtering by GPU nodes."""
        raise NotImplementedError()

    async def get_kubernetes_nodes(self) -> List[Dict]:
        """
        Get all Kubernetes nodes via k8s client, optionally filtering by GPU nodes.
        """
        nodes = []
        try:
            node_list = self._get_nodes()
            for node in node_list.items:
                nodes.append(self._extract_node_info(node))
        except Exception as e:
            logger.error(f"Failed to get Kubernetes nodes: {e}")
            raise
        return nodes

    @abc.abstractmethod
    def get_node(self, name: str) -> V1Node:
        """
        Retrieve all node from the environment by name
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_nodes(self) -> V1NodeList:
        """
        Retrieve all nodes from the environment
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def patch_node(self, name: str, body: Dict) -> V1Node:
        raise NotImplementedError()

    def _extract_node_info(self, node: V1Node):
        gpu_count = int(node.status.capacity["nvidia.com/gpu"])
        gpu_mem_mb = int(node.metadata.labels.get("nvidia.com/gpu.memory", "32"))
        gpu_mem_gb = int(gpu_mem_mb / 1024)
        cpu_count = (
            int(node.status.capacity["cpu"]) - 2
        )  # leave 2 CPUs for incidentals, daemon sets, etc.
        cpus_per_gpu = 1 if cpu_count <= gpu_count else min(4, math.floor(cpu_count / gpu_count))
        raw_mem = node.status.capacity["memory"]
        if raw_mem.endswith("Ki"):
            total_memory_gb = int(int(raw_mem.replace("Ki", "")) / 1024 / 1024) - 6
        elif raw_mem.endswith("Mi"):
            total_memory_gb = int(int(raw_mem.replace("Mi", "")) / 1024) - 6
        elif raw_mem.endswith("Gi"):
            total_memory_gb = int(raw_mem.replace("Gi", "")) - 6
        memory_gb_per_gpu = (
            1
            if total_memory_gb <= gpu_count
            else min(gpu_mem_gb, math.floor(total_memory_gb * 0.8 / gpu_count))
        )
        node_info = {
            "name": node.metadata.name,
            "validator": node.metadata.labels.get("chutes/validator"),
            "server_id": node.metadata.uid,
            "status": node.status.phase,
            "ip_address": node.metadata.labels.get("chutes/external-ip"),
            "cpu_per_gpu": cpus_per_gpu,
            "memory_gb_per_gpu": memory_gb_per_gpu,
        }
        return node_info

    @abc.abstractmethod
    async def get_deployment(self, deployment_id: str) -> Dict:
        """Get a single deployment by ID."""
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_deployment(self, name: str, namespace: str = settings.namespace):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_deployments(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ) -> V1DeploymentList:
        """
        Get deployments, optinally filtering by namespace and labels
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def watch_deployments(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ) -> Generator[WatchEvent, None, None]:
        """
        Watch deployments and yield events with type and deployment object.

        Yields:
            dict: Event with 'type' ('ADDED', 'MODIFIED', 'DELETED') and 'object' (V1Deployment)
        """
        raise NotImplementedError()

    def _is_deployment_ready(self, deployment):
        """
        Check if a deployment is "ready"
        """
        return (
            deployment.status.available_replicas is not None
            and deployment.status.available_replicas == deployment.spec.replicas
            and deployment.status.ready_replicas == deployment.spec.replicas
            and deployment.status.updated_replicas == deployment.spec.replicas
        )

    async def get_deployed_chutes(self) -> List[Dict]:
        """Get all chutes deployments from kubernetes."""
        deployments = []
        deployments_list = self.get_deployments(
            namespace=settings.namespace, label_selector={"chutes/chute": "true"}
        )
        for deployment in deployments_list.items:
            deployments.append(self._extract_deployment_info(deployment))
            logger.info(
                f"Found chute deployment: {deployment.metadata.name} in namespace {deployment.metadata.namespace}"
            )
        return deployments

    async def delete_code(self, chute_id: str, version: str) -> None:
        """
        Delete the code configmap associated with a chute & version.
        """
        try:
            code_uuid = self._get_code_uuid(chute_id, version)
            self.delete_config_map(f"{CHUTE_CODE_CM_PREFIX}-{code_uuid}")
        except ApiException as exc:
            if exc.status != 404:
                logger.error(f"Failed to delete code reference: {exc}")
                raise

    @abc.abstractmethod
    def delete_config_map(self, name: str, namespace=settings.namespace):
        raise NotImplementedError()

    @lru_cache(maxsize=5)
    def _get_code_uuid(self, chute_id: str, version: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chute_id}::{version}"))

    async def wait_for_deletion(self, label_selector: str, timeout_seconds: int = 120) -> None:
        """
        Wait for a deleted pod to be fully removed.
        """
        pods = self._get_pods(settings.namespace, label_selector)
        if not pods.items:
            logger.info(f"Nothing to wait for: {label_selector}")
            return

        # w = watch.Watch()
        try:
            for event in self.watch_pods(
                namespace=settings.namespace, label_selector=label_selector, timeout=timeout_seconds
            ):
                if event.is_deleted:
                    logger.success(f"Deletion of {label_selector=} is complete")
                    break
        except Exception as exc:
            logger.warning(f"Error waiting for pods to be deleted: {exc}")

    async def undeploy(self, deployment_id: str) -> None:
        """Delete a chute, and associated service."""
        try:
            self.delete_service(f"{CHUTE_SVC_PREFIX}-{deployment_id}")
        except Exception as exc:
            logger.warning(f"Error deleting deployment service from k8s: {exc}")

        try:
            self.delete_deployment(f"{CHUTE_DEPLOY_PREFIX}-{deployment_id}")
        except Exception as exc:
            logger.warning(f"Error deleting deployment from k8s: {exc}")
        await self.wait_for_deletion(f"chutes/deployment-id={deployment_id}", timeout_seconds=15)

    @abc.abstractmethod
    def delete_service(self, name, namespace=settings.namespace):
        raise NotImplementedError()

    async def create_code_config_map(self, chute: Chute) -> None:
        """Create a ConfigMap to store the chute code."""
        try:
            config_map = self._create_code_config_map(chute)
            self.create_config_map(config_map)
        except ApiException as e:
            if e.status != 409:
                raise

    def _create_code_config_map(self, chute: Chute) -> V1ConfigMap:
        code_uuid = self._get_code_uuid(chute.chute_id, chute.version)
        config_map = V1ConfigMap(
            metadata=V1ObjectMeta(
                name=f"{CHUTE_CODE_CM_PREFIX}-{code_uuid}",
                labels={
                    "chutes/chute-id": chute.chute_id,
                    "chutes/version": chute.version,
                    "chutes/code": "true",
                },
            ),
            data={chute.filename: chute.code},
        )
        return config_map

    @abc.abstractmethod
    def create_config_map(self, config_map: V1ConfigMap, namespace=settings.namespace):
        raise NotImplementedError()

    async def deploy_chute(
        self, chute_id: Union[str | Chute], server_id: Union[str | Server]
    ) -> Tuple[Deployment, Any, Any]:
        """Deploy a chute!"""
        # Backwards compatible types...
        if isinstance(chute_id, Chute):
            chute_id = chute_id.chute_id
        if isinstance(server_id, Server):
            server_id = server_id.server_id

        async with get_session() as session:
            chute = await self._get_chute(session, chute_id)
            server = await self._get_server(session, server_id)
            available_gpus = self._verify_gpus(chute, server)
            deployment_id = await self._track_deployment(session, chute, server, available_gpus)

        # Create the deployment.
        deployment = build_chute_deployment(deployment_id, chute, server)

        # And the service that exposes it.
        service = build_chute_service(deployment_id, chute)

        return await self._deploy_chute(deployment_id, deployment, service, chute, server)

    async def _get_chute(self, session: AsyncSession, chute_id: str):
        chute = (
            (await session.execute(select(Chute).where(Chute.chute_id == chute_id)))
            .unique()
            .scalar_one_or_none()
        )

        if not chute:
            raise DeploymentFailure(f"Failed to find chute: {chute_id=}")

        return chute

    async def _get_server(self, session: AsyncSession, server_id: str):
        server = (
            (await session.execute(select(Server).where(Server.server_id == server_id)))
            .unique()
            .scalar_one_or_none()
        )
        if not server:
            raise DeploymentFailure(f"Failed to find server: {server_id=}")

        return server

    def _verify_gpus(self, chute: Chute, server: Server):
        # Make sure the node has capacity.
        gpus_allocated = 0
        available_gpus = {gpu.gpu_id for gpu in server.gpus if gpu.verified}
        for deployment in server.deployments:
            gpus_allocated += len(deployment.gpus)
            available_gpus -= {gpu.gpu_id for gpu in deployment.gpus}
        if len(available_gpus) - chute.gpu_count < 0:
            raise DeploymentFailure(
                f"Server {server.server_id} name={server.name} cannot allocate {chute.gpu_count} GPUs, already using {gpus_allocated} of {len(server.gpus)}"
            )
        return available_gpus

    async def _track_deployment(
        self, session: AsyncSession, chute: Chute, server: Server, available_gpus
    ):
        # Immediately track this deployment (before actually creating it) to avoid allocation contention.
        deployment_id = str(uuid.uuid4())
        gpus = list([gpu for gpu in server.gpus if gpu.gpu_id in available_gpus])[: chute.gpu_count]
        deployment = Deployment(
            deployment_id=deployment_id,
            server_id=server.server_id,
            validator=server.validator,
            chute_id=chute.chute_id,
            version=chute.version,
            active=False,
            verified_at=None,
            stub=True,
        )
        session.add(deployment)
        deployment.gpus = gpus
        await session.commit()

        return deployment_id

    @abc.abstractmethod
    async def _deploy_chute(
        self,
        deployment_id: str,
        deployment: V1Deployment,
        service: V1Service,
        chute: Chute,
        server: Server,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    async def deploy_graval(
        node: V1Node, deployment: V1Deployment, service: V1Service
    ) -> Tuple[V1Deployment, V1Service]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def cleanup_graval(self, node: V1Node):
        raise NotImplementedError()


# Legacy single-cluster implementation
class SingleClusterK8sOperator(K8sOperator):
    """Kubernetes operations for legacy single-cluster setup."""

    def get_node(self, name: str) -> V1Node:
        """
        Retrieve all nodes from the environment
        """
        return k8s_core_client().read_node(name=name)

    def _get_nodes(self):
        """
        Retrieve all nodes from the environment
        """
        node_list = k8s_core_client().list_node(field_selector=None, label_selector="chutes/worker")
        return node_list

    def patch_node(self, name, body):
        return k8s_core_client().patch_node(name=name, body=body)

    def watch_pods(self, namespace=None, label_selector=None, field_selector=None, timeout=120):
        if label_selector:
            label_selector = (
                label_selector
                if isinstance(label_selector, str)
                else ",".join(f"{k}={v}" for k, v in label_selector.items())
            )

        if field_selector:
            field_selector = (
                field_selector
                if isinstance(field_selector, str)
                else ",".join(f"{k}={v}" for k, v in field_selector.items())
            )

        # Use the standard Kubernetes watch mechanism
        for event in watch.Watch().stream(
            k8s_core_client().list_namespaced_pod,
            namespace=namespace,
            label_selector=label_selector,
            field_selector=field_selector,
            timeout_seconds=timeout,
        ):
            yield WatchEvent.from_dict(event)

    def _get_pods(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ) -> V1PodList:
        if label_selector:
            label_selector = (
                label_selector
                if isinstance(label_selector, str)
                else ",".join([f"{k}={v}" for k, v in label_selector.items()])
            )

        if field_selector:
            field_selector = (
                field_selector
                if isinstance(field_selector, str)
                else ",".join([f"{k}={v}" for k, v in field_selector.items()])
            )

        pods = k8s_core_client().list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
            field_selector=field_selector,
            timeout_seconds=timeout,
        )
        return pods

    async def get_deployment(self, deployment_id: str) -> Dict:
        """
        Get a single deployment by ID.
        """
        deployment = k8s_app_client().read_namespaced_deployment(
            namespace=settings.namespace,
            name=f"chute-{deployment_id}",
        )
        return self._extract_deployment_info(deployment)

    def get_deployments(
        self,
        namespace: Optional[str] = settings.namespace,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ) -> V1DeploymentList:
        """
        Get deployment, optinally filtering by namespace and labels
        """
        if label_selector:
            label_selector = (
                label_selector
                if isinstance(label_selector, str)
                else ",".join(f"{k}={v}" for k, v in label_selector.items())
            )

        if field_selector:
            field_selector = (
                field_selector
                if isinstance(field_selector, str)
                else ",".join(f"{k}={v}" for k, v in field_selector.items())
            )

        deployment_list = k8s_app_client().list_namespaced_deployment(
            namespace=namespace,
            label_selector=label_selector,
            field_selector=field_selector,
            timeout_seconds=timeout,
        )

        return deployment_list

    def watch_deployments(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ):
        """Watch deployments using standard Kubernetes watch API."""
        if label_selector:
            label_selector = (
                label_selector
                if isinstance(label_selector, str)
                else ",".join(f"{k}={v}" for k, v in label_selector.items())
            )

        if field_selector:
            field_selector = (
                field_selector
                if isinstance(field_selector, str)
                else ",".join(f"{k}={v}" for k, v in field_selector.items())
            )

        # Use the standard Kubernetes watch mechanism
        for event in watch.Watch().stream(
            k8s_app_client().list_namespaced_deployment,
            namespace=namespace,
            label_selector=label_selector,
            field_selector=field_selector,
            timeout_seconds=timeout,
        ):
            yield WatchEvent.from_dict(event)

    def delete_deployment(self, name, namespace=settings.namespace):
        k8s_app_client().delete_namespaced_deployment(
            name=name,
            namespace=namespace,
        )

    def delete_service(self, name, namespace=settings.namespace):
        k8s_core_client().delete_namespaced_service(
            name=name,
            namespace=namespace,
        )

    def delete_config_map(self, name, namespace=settings.namespace):
        k8s_core_client().delete_namespaced_config_map(name=name, namespace=namespace)

    def create_config_map(self, config_map: V1ConfigMap, namespace=settings.namespace):
        k8s_core_client().create_namespaced_config_map(namespace=namespace, body=config_map)

    async def _deploy_chute(
        self,
        deployment_id: str,
        deployment: V1Deployment,
        service: V1Service,
        chute: Chute,
        server: Server,
    ):
        try:
            created_service = k8s_core_client().create_namespaced_service(
                namespace=settings.namespace, body=service
            )
            created_deployment = k8s_app_client().create_namespaced_deployment(
                namespace=settings.namespace, body=deployment
            )
            deployment_port = created_service.spec.ports[0].node_port
            async with get_session() as session:
                deployment = (
                    (
                        await session.execute(
                            select(Deployment).where(Deployment.deployment_id == deployment_id)
                        )
                    )
                    .unique()
                    .scalar_one_or_none()
                )
                if not deployment:
                    raise DeploymentFailure("Deployment disappeared mid-flight!")
                deployment.host = server.ip_address
                deployment.port = deployment_port
                deployment.stub = False
                await session.commit()
                await session.refresh(deployment)

            return deployment, created_deployment, created_service
        except ApiException as exc:
            try:
                k8s_core_client().delete_namespaced_service(
                    name=f"{CHUTE_SVC_PREFIX}-{deployment_id}",
                    namespace=settings.namespace,
                )
            except Exception:
                ...
            try:
                k8s_core_client().delete_namespaced_deployment(
                    name=f"chute-{deployment_id}",
                    namespace=settings.namespace,
                )
            except Exception:
                ...
            raise DeploymentFailure(
                f"Failed to deploy chute {chute.chute_id} with version {chute.version}: {exc}\n{traceback.format_exc()}"
            )

    async def deploy_graval(self, node: V1Node, deployment: V1Deployment, service: V1Service):
        try:
            created_service = k8s_core_client().create_namespaced_service(
                namespace=settings.namespace, body=service
            )
            created_deployment = k8s_app_client().create_namespaced_deployment(
                namespace=settings.namespace, body=deployment
            )

            # Track the verification port.
            expected_port = created_service.spec.ports[0].node_port
            async with get_session() as session:
                result = await session.execute(
                    update(Server)
                    .where(Server.server_id == node.metadata.uid)
                    .values(verification_port=created_service.spec.ports[0].node_port)
                    .returning(Server.verification_port)
                )
                port = result.scalar_one_or_none()
                if port != expected_port:
                    raise DeploymentFailure(
                        f"Unable to track verification port for newly added node: {expected_port=} actual_{port=}"
                    )
                await session.commit()
            return created_deployment, created_service
        except ApiException as exc:
            try:
                k8s_core_client().delete_namespaced_service(
                    name=service.metadata.name,
                    namespace=settings.namespace,
                )
            except Exception:
                ...
            try:
                k8s_core_client().delete_namespaced_deployment(
                    name=deployment.metadata.name,
                    namespace=settings.namespace,
                )
            except Exception:
                ...
            raise DeploymentFailure(
                f"Failed to deploy GraVal: {str(exc)}:\n{traceback.format_exc()}"
            )

    async def cleanup_graval(self, node: V1Node):
        node_name = node.metadata.name
        nice_name = node_name.replace(".", "-")
        try:
            self.delete_service(f"{GRAVAL_SVC_PREFIX}-{nice_name}")
        except Exception:
            ...

        try:
            self.delete_deployment(f"{GRAVAL_DEPLOY_PREFIX}-{nice_name}")
            label_selector = f"graval-node={nice_name}"

            await self.wait_for_deletion(label_selector)
        except Exception:
            ...


class KarmadaK8sOperator(K8sOperator):
    """Kubernetes operations for Karmada-based multi-cluster setup."""

    # This class will implement the K8sOperator interface but translate operations
    # to work with Karmada's multi-cluster orchestration

    @property
    def karmada_api_client(self):
        return k8s_api_client(karmada_api=True)

    @property
    def karmada_custom_objects_client(self):
        return k8s_custom_objects_client(karmada_api=True)

    @property
    def karmada_app_client(self):
        return k8s_app_client(True)

    @property
    def karmada_core_client(self):
        return k8s_core_client(True)

    def get_node(self, name: str) -> V1Node:
        """
        Retrieve all nodes from the environment
        """
        node_path = f"{SEARCH_NODES_PATH}/{name}"
        response = self._search(node_path)
        node_list = self.karmada_api_client.deserialize(ApiResponse(response), "V1NodeList")
        return node_list.items[0] if len(node_list.items) == 1 else None

    def _get_nodes(self) -> V1NodeList:
        response = self._search(SEARCH_NODES_PATH)
        node_list = self.karmada_api_client.deserialize(ApiResponse(response), "V1NodeList")
        return node_list

    def patch_node(self, name: str, body: Dict) -> V1Node:
        # Construct the API path for the Search API cache endpoint
        api_path = f"/apis/cluster.karmada.io/v1alpha1/clusters/{name}/proxy/api/v1/nodes/{name}"

        headers = {"Content-Type": "application/strategic-merge-patch+json"}

        # Call the Search API using the client
        logger.info("Patching node via karmada proxy")
        self.karmada_api_client.call_api(api_path, "PATCH", header_params=headers, body=body)

        return self.get_node(name)

    async def get_deployment(self, deployment_id: str) -> Dict:
        """Get a single deployment by ID."""
        deployment_name = f"{CHUTE_DEPLOY_PREFIX}-{deployment_id}"
        deploy_list = self.get_deployments(field_selector=f"metadata.name={deployment_name}")

        if len(deploy_list.items) == 0:
            logger.warning(f"Failed to find deployment {deployment_name}")
            raise ApiException(status=404, reason=f"Failed to find deployment {deployment_name}")

        # Handle case where no deployments found or more than one found
        return self._extract_deployment_info(deploy_list.items[0])

    def delete_deployment(self, name, namespace=settings.namespace):
        self.karmada_app_client.delete_namespaced_deployment(
            name=name,
            namespace=namespace,
        )

    def delete_service(self, name, namespace=settings.namespace):
        self.karmada_core_client.delete_namespaced_service(
            name=name,
            namespace=namespace,
        )

    def delete_config_map(self, name, namespace=settings.namespace):
        self.karmada_core_client.delete_namespaced_config_map(name=name, namespace=namespace)

    def create_config_map(self, config_map: V1ConfigMap, namespace=settings.namespace):
        # This is the one exception where we do not create the propagation policy for
        # the CM since we use a predefined PP to propagate all chute code CMs to all clusters
        self.karmada_core_client.create_namespaced_config_map(namespace=namespace, body=config_map)

    async def undeploy(self, deployment_id: str) -> None:
        """Delete a deployment, and associated service."""
        await super().undeploy(deployment_id)
        chute_pp_name = f"{CHUTE_PP_PREFIX}-{deployment_id}"
        self._delete_propagation_policy(chute_pp_name)

    async def _deploy_chute(
        self,
        deployment_id: str,
        deployment: V1Deployment,
        service: V1Service,
        chute: Chute,
        server: Server,
    ):
        try:
            created_service = self.karmada_core_client.create_namespaced_service(
                namespace=settings.namespace, body=service
            )
            created_deployment = self.karmada_app_client.create_namespaced_deployment(
                namespace=settings.namespace, body=deployment
            )
            self._create_chute_propagation_policy(deployment_id, service, deployment, server)

            deployment_port = created_service.spec.ports[0].node_port
            async with get_session() as session:
                deployment = (
                    (
                        await session.execute(
                            select(Deployment).where(Deployment.deployment_id == deployment_id)
                        )
                    )
                    .unique()
                    .scalar_one_or_none()
                )
                if not deployment:
                    raise DeploymentFailure("Deployment disappeared mid-flight!")
                deployment.host = server.ip_address
                deployment.port = deployment_port
                deployment.stub = False
                await session.commit()
                await session.refresh(deployment)

            return deployment, created_deployment, created_service
        except ApiException as exc:
            self._cleanup_chute_resources(deployment_id)

            raise DeploymentFailure(
                f"Failed to deploy chute {chute.chute_id} with version {chute.version}: {exc}\n{traceback.format_exc()}"
            )

    def _cleanup_chute_resources(self, deployment_id: str):
        try:
            self.delete_service(f"{CHUTE_SVC_PREFIX}-{deployment_id}")
        except Exception:
            ...
        try:
            self.delete_deployment(f"{CHUTE_DEPLOY_PREFIX}-{deployment_id}")
        except Exception:
            ...
        try:
            self._delete_propagation_policy(f"{CHUTE_PP_PREFIX}-{deployment_id}")
        except Exception:
            ...

    def _create_chute_propagation_policy(
        self,
        deployment_id: str,
        service: V1Service,
        deployment: V1Deployment,
        server: Server,
    ):
        pp = PropagationPolicy(
            name=f"{CHUTE_PP_PREFIX}-{deployment_id}",
            namespace=settings.namespace,
            resource_selectors=[
                ResourceSelector(
                    api_version="apps/v1", kind="Deployment", name=deployment.metadata.name
                ),
                ResourceSelector(api_version="v1", kind="Service", name=service.metadata.name),
            ],
            placement=Placement(
                cluster_affinity=ClusterAffinity(cluster_names=[server.name]),
                replica_scheduling=ReplicaScheduling(scheduling_type="Duplicated"),
            ),
        )
        self._create_propagation_policy(pp)

    def _create_graval_propagation_policy(
        self, node: V1Node, service: V1Service, deployment: V1Deployment
    ):
        pp = PropagationPolicy(
            name=f"{GRAVAL_PP_PREFIX}-{node.metadata.name.replace('.', '-')}",
            namespace=settings.namespace,
            resource_selectors=[
                ResourceSelector(
                    api_version="apps/v1", kind="Deployment", name=deployment.metadata.name
                ),
                ResourceSelector(api_version="v1", kind="Service", name=service.metadata.name),
            ],
            placement=Placement(
                cluster_affinity=ClusterAffinity(cluster_names=[node.metadata.name]),
                replica_scheduling=ReplicaScheduling(scheduling_type="Duplicated"),
            ),
        )
        self._create_propagation_policy(pp)

    def _create_propagation_policy(
        self, propagation_policy: PropagationPolicy, namespace: str = settings.namespace
    ):
        self.karmada_custom_objects_client.create_namespaced_custom_object(
            group="policy.karmada.io",
            version="v1alpha1",
            namespace=namespace,
            plural="propagationpolicies",
            body=propagation_policy.to_dict(),
        )

    def _delete_propagation_policy(self, pp_name, namespace=settings.namespace):
        self.karmada_custom_objects_client.delete_namespaced_custom_object(
            group="policy.karmada.io",
            version="v1alpha1",
            namespace=namespace,
            plural="propagationpolicies",
            name=pp_name,
        )

    def watch_pods(self, namespace=None, label_selector=None, field_selector=None, timeout=120):
        for event in self._watch_resources(
            timeout,
            self._get_pods,
            namespace=namespace,
            label_selector=label_selector,
            field_selector=field_selector,
        ):
            yield event

    def _get_pods(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ) -> V1PodList:
        response = self._search(SEARCH_PODS_PATH)
        pod_list = self.karmada_api_client.deserialize(ApiResponse(response), "V1PodList")

        pod_list.items[:] = [
            pod
            for pod in pod_list.items
            if self._matches_filters(pod, namespace, label_selector, field_selector)
        ]

        return pod_list

    def get_deployments(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
        timeout=120,
    ):
        response = self._search(SEARCH_DEPLOYMENTS_PATH)
        deploy_list = self.karmada_api_client.deserialize(ApiResponse(response), "V1DeploymentList")

        deploy_list.items[:] = [
            deployment
            for deployment in deploy_list.items
            if self._matches_filters(deployment, namespace, label_selector, field_selector)
        ]

        return deploy_list

    def watch_deployments(
        self, namespace=None, label_selector=None, field_selector=None, timeout=120
    ):
        for event in self._watch_resources(
            timeout,
            self.get_deployments,
            namespace=namespace,
            label_selector=label_selector,
            field_selector=field_selector,
        ):
            yield event

    def _watch_resources(self, timeout, search_func, *search_args, **search_kwargs):
        previous_state = {}

        start_time = time.time()

        while True:
            try:
                # Get current deployments from search API
                current_list = search_func(*search_args, **search_kwargs)

                current_state = {resource.metadata.uid: resource for resource in current_list.items}

                # Find added deployments
                for uid, resource in current_state.items():
                    if uid not in previous_state:
                        yield WatchEvent(type=WatchEventType.ADDED, object=resource)
                    elif (
                        previous_state[uid].metadata.resource_version
                        != resource.metadata.resource_version
                    ):
                        yield WatchEvent(type=WatchEventType.MODIFIED, object=resource)

                # Find deleted deployments
                for uid, resource in previous_state.items():
                    if uid not in current_state:
                        yield WatchEvent(type=WatchEventType.DELETED, object=resource)

                # Update previous state
                previous_state = current_state

                # Check timeout
                if time.time() - start_time >= timeout:
                    break

                time.sleep(2)

            except Exception as e:
                logger.error(f"Error in watch_deployments: {e}")
                break

    def _matches_filters(
        self,
        deployment: Union[V1Pod | V1Deployment],
        namespace: Optional[str] = None,
        label_selector: Optional[Union[str | Dict[str, str]]] = None,
        field_selector: Optional[Union[str | Dict[str, str]]] = None,
    ) -> bool:
        """Check if deployment matches all specified filters."""

        # Namespace filter
        if namespace and deployment.metadata.namespace != namespace:
            return False

        # Label selector filter
        if label_selector:
            deployment_labels = deployment.metadata.labels or {}

            if isinstance(label_selector, str):
                # Parse string label selector (e.g., "app=nginx,version=1.0")
                label_dict = self._parse_label_selector(label_selector)
            else:
                label_dict = label_selector

            # Check if all required labels match
            for key, value in label_dict.items():
                if deployment_labels.get(key) != value:
                    return False

        # Field selector filter (example implementation for common fields)
        if field_selector:
            if isinstance(field_selector, str):
                field_dict = self._parse_field_selector(field_selector)
            else:
                field_dict = field_selector

            for field_path, expected_value in field_dict.items():
                actual_value = self._get_field_value(deployment, field_path)
                if actual_value != expected_value:
                    return False

        return True

    def _parse_label_selector(self, selector: str) -> Dict[str, str]:
        """Parse label selector string into dictionary."""
        labels = {}
        if selector:
            for pair in selector.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    labels[key.strip()] = value.strip()
        return labels

    def _parse_field_selector(self, selector: str) -> Dict[str, str]:
        """Parse field selector string into dictionary."""
        fields = {}
        if selector:
            for pair in selector.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    fields[key.strip()] = value.strip()
        return fields

    def _get_field_value(self, deployment, field_path: str):
        """Get field value from deployment using dot notation."""
        obj = deployment
        for part in field_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    def _search(self, api_path, query_params={}):
        """
        Search using the Karamada Search API
        """

        response = self.karmada_api_client.call_api(
            api_path,
            "GET",
            query_params=query_params,
            response_type="object",
            _return_http_data_only=True,
        )

        return response

    async def deploy_graval(self, node: V1Node, deployment: V1Deployment, service: V1Service):
        try:
            created_service = self.karmada_core_client.create_namespaced_service(
                namespace=settings.namespace, body=service
            )
            created_deployment = self.karmada_app_client.create_namespaced_deployment(
                namespace=settings.namespace, body=deployment
            )

            # Create propagation policy
            self._create_graval_propagation_policy(node, service, deployment)

            # Track the verification port.
            # Need to get the service port from the search API
            expected_port = created_service.spec.ports[0].node_port
            async with get_session() as session:
                result = await session.execute(
                    update(Server)
                    .where(Server.server_id == node.metadata.uid)
                    .values(verification_port=created_service.spec.ports[0].node_port)
                    .returning(Server.verification_port)
                )
                port = result.scalar_one_or_none()
                if port != expected_port:
                    raise DeploymentFailure(
                        f"Unable to track verification port for newly added node: {expected_port=} actual_{port=}"
                    )
                await session.commit()
            return created_deployment, created_service
        except ApiException as exc:
            self._cleanup_graval_deployment(node, service, deployment)
            raise DeploymentFailure(
                f"Failed to deploy GraVal: {str(exc)}:\n{traceback.format_exc()}"
            )

    async def cleanup_graval(self, node: V1Node):
        node_name = node.metadata.name
        nice_name = node_name.replace(".", "-")
        try:
            self.delete_service(f"{GRAVAL_SVC_PREFIX}-{nice_name}")
        except Exception:
            ...

        try:
            self.delete_deployment(f"{GRAVAL_DEPLOY_PREFIX}-{nice_name}")
            label_selector = f"graval-node={nice_name}"

            await self.wait_for_deletion(label_selector)
        except Exception:
            ...

        try:
            self._delete_propagation_policy(f"{GRAVAL_PP_PREFIX}-{nice_name}")
        except Exception:
            ...

    def _cleanup_graval_deployment(
        self, node: V1Node, service: V1Service, deployment: V1Deployment
    ):
        try:
            self.delete_service(service.metadata.name)
        except Exception:
            ...
        try:
            self.delete_deployment(deployment.metadata.name)
        except Exception:
            ...
        try:
            self._delete_propagation_policy(
                f"{GRAVAL_PP_PREFIX}-{node.metadata.name.replace('.', '-')}"
            )
        except Exception:
            ...
