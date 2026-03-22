import os
from functools import lru_cache
from typing import Optional
from chutes_miner.api.config import k8s_app_client, k8s_batch_client, k8s_core_client
from chutes_miner.api.k8s.config import KubeConfig, MultiClusterKubeConfig
from chutes_miner.api.k8s.exceptions import KubeContextNotFound, KubeconfigNotFound
from kubernetes import client, config
from loguru import logger

CONTROL_PLANE_LABEL = "node-role.kubernetes.io/control-plane"


@lru_cache(maxsize=1)
def _get_local_cluster_context() -> Optional[str]:
    """
    Return the context name for the cluster we're running in (Scenario 3: control-as-GPU).
    Inferred from the control-plane node name when running in-cluster.
    Cached for process lifetime; node name does not change at runtime.
    """
    result = None
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        try:
            core = k8s_core_client()
            nodes = core.list_node(label_selector=CONTROL_PLANE_LABEL)
            if nodes.items:
                result = nodes.items[0].metadata.name
        except Exception:
            pass
    return result


class KubernetesMultiClusterClientManager:
    def __init__(self, default_timeout: int = 30):
        """
        Args:
            default_timeout: Default timeout in seconds for API calls (default: 10)
        """
        self.multi_config = MultiClusterKubeConfig()
        self.default_timeout = default_timeout

    def _use_incluster_for_context(self, context_name: str) -> bool:
        """True when context refers to our cluster (Scenario 3: control-as-GPU)."""
        local = _get_local_cluster_context()
        return local is not None and context_name == local

    def get_api_client(self, context_name, timeout: Optional[int] = None) -> client.ApiClient:
        if self._use_incluster_for_context(context_name):
            return k8s_core_client().api_client
        _client = None
        try:
            _client = self._get_client_for_context(context_name, timeout=timeout)
        except KubeContextNotFound as e:
            logger.error(f"Failed to get api client:\n{e}")

        return _client

    def get_app_client(self, context_name, timeout: Optional[int] = None) -> client.AppsV1Api:
        if self._use_incluster_for_context(context_name):
            return k8s_app_client()
        _client = None
        try:
            api_client = self._get_client_for_context(context_name, timeout=timeout)
            _client = client.AppsV1Api(api_client)
        except KubeContextNotFound as e:
            logger.error(f"Failed to get app client:\n{e}")

        return _client

    def get_core_client(
        self,
        context_name: str,
        kubeconfig: Optional[KubeConfig] = None,
        timeout: Optional[int] = None,
    ) -> client.CoreV1Api:
        if self._use_incluster_for_context(context_name):
            return k8s_core_client()
        _client = None
        try:
            api_client = self._get_client_for_context(context_name, kubeconfig, timeout=timeout)
            _client = client.CoreV1Api(api_client)
        except KubeContextNotFound as e:
            logger.error(f"Failed to get core client:\n{e}")

        return _client

    def get_batch_client(
        self, context_name: str, timeout: Optional[int] = None
    ) -> client.BatchV1Api:
        if self._use_incluster_for_context(context_name):
            return k8s_batch_client()
        _client = None
        try:
            api_client = self._get_client_for_context(context_name, timeout=timeout)
            _client = client.BatchV1Api(api_client)
        except KubeContextNotFound as e:
            logger.error(f"Failed to get batch client:\n{e}")

        return _client

    @lru_cache(maxsize=10)
    def _get_client_for_context(
        self, context: str, kubeconfig: Optional[KubeConfig] = None, timeout: Optional[int] = None
    ) -> client.ApiClient:
        """Create a new client configured for the specified context

        Args:
            context: The kubernetes context name
            kubeconfig: Optional kubeconfig override
            timeout: Optional timeout in seconds (uses default_timeout if not specified)
        """
        _kubeconfig = kubeconfig if kubeconfig else self.multi_config.kubeconfig

        if not _kubeconfig:
            raise KubeconfigNotFound(
                "No kubeconfig currently loaded and no override kubeconfig provided"
            )

        if context not in [c.name for c in _kubeconfig.contexts]:
            raise KubeContextNotFound(f"Context {context} does not exist in kubeconfig.")

        # Create configuration for this specific context
        api_client = config.kube_config.new_client_from_config_dict(
            _kubeconfig.to_dict(), context=context, persist_config=False
        )

        # Set the timeout on the configuration
        _timeout = timeout if timeout is not None else self.default_timeout
        api_client.configuration.timeout = _timeout

        return api_client
