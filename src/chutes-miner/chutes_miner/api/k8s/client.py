from functools import lru_cache
from typing import Optional
from chutes_miner.api.k8s.config import KubeConfig, MultiClusterKubeConfig
from chutes_miner.api.k8s.exceptions import KubeContextNotFound, KubeconfigNotFound
from kubernetes import client, config
from loguru import logger


class KubernetesMultiClusterClientManager:
    def __init__(self):
        self.multi_config = MultiClusterKubeConfig()

    def get_api_client(self, context_name) -> client.ApiClient:
        _client = None
        try:
            _client = self._get_client_for_context(context_name)
        except (KubeContextNotFound, KubeContextNotFound) as e:
            logger.error(f"Failed to get api client:\n{e}")

        return _client

    def get_app_client(self, context_name) -> client.AppsV1Api:
        _client = None
        try:
            api_client = self._get_client_for_context(context_name)
            _client = client.AppsV1Api(api_client)
        except (KubeContextNotFound, KubeContextNotFound) as e:
            logger.error(f"Failed to get app client:\n{e}")

        return _client

    def get_core_client(
        self, context_name: str, kubeconfig: Optional[KubeConfig] = None
    ) -> client.CoreV1Api:
        _client = None
        try:
            api_client = self._get_client_for_context(context_name, kubeconfig)
            _client = client.CoreV1Api(api_client)
        except (KubeContextNotFound, KubeContextNotFound) as e:
            logger.error(f"Failed to get core client:\n{e}")

        return _client

    def get_batch_client(self, context_name: str) -> client.BatchV1Api:
        _client = None
        try:
            api_client = self._get_client_for_context(context_name)
            _client = client.BatchV1Api(api_client)
        except (KubeContextNotFound, KubeContextNotFound) as e:
            logger.error(f"Failed to get batch client:\n{e}")

        return _client

    @lru_cache(maxsize=10)
    def _get_client_for_context(
        self, context: str, kubeconfig: Optional[KubeConfig] = None
    ) -> client.ApiClient:
        """Create a new client configured for the specified context"""

        _kubeconfig = kubeconfig if kubeconfig else self.multi_config.kubeconfig

        if not _kubeconfig:
            raise KubeconfigNotFound(
                "No kubeconfig currently loaded and no override kubeconfig provided"
            )

        if context not in [c.name for c in _kubeconfig.contexts]:
            raise KubeContextNotFound(f"Context {context} does not exist in kubeconfig.")

        # Create configuration for this specific context
        return config.kube_config.new_client_from_config_dict(
            _kubeconfig.to_dict(), context=context, persist_config=False
        )
