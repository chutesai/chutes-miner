import os
from unittest.mock import MagicMock, patch

from chutes_miner.api.k8s.client import (
    KubernetesMultiClusterClientManager,
    _get_local_cluster_context,
)
from chutes_miner.api.k8s.config import KubeConfig
from kubernetes import client
import yaml


def test_get_core_client(sample_kubeconfig):
    
    with patch('chutes_miner.api.k8s.config.MultiClusterKubeConfig._load') as mock_load:
        kubeconfig_dict = yaml.safe_load(sample_kubeconfig["kubeconfig"])
        kubeconfig = KubeConfig.from_dict(kubeconfig_dict)

        manager = KubernetesMultiClusterClientManager()

        _client = manager.get_core_client("chutes-miner-gpu-0", kubeconfig)

        assert type(_client) == client.CoreV1Api


def test_get_local_cluster_context_not_in_cluster():
    """When not in a pod, _get_local_cluster_context returns None."""
    with patch.dict(os.environ, {}, clear=False):
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            del os.environ["KUBERNETES_SERVICE_HOST"]
    result = _get_local_cluster_context()
    assert result is None


def test_get_local_cluster_context_auto_detect_from_control_plane():
    """When in-cluster, infer local context from control-plane node name."""
    mock_node = MagicMock()
    mock_node.metadata.name = "control-node-01"
    mock_node_list = MagicMock()
    mock_node_list.items = [mock_node]

    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=False):
        with patch("chutes_miner.api.k8s.client.k8s_core_client") as mock_core:
            mock_core.return_value.list_node.return_value = mock_node_list
            result = _get_local_cluster_context()
    assert result == "control-node-01"


def test_use_incluster_returns_false_when_not_local(mock_get_local_cluster_context):
    """When context does not match local cluster, use kubeconfig."""
    # mock_get_local_cluster_context defaults to None
    manager = KubernetesMultiClusterClientManager()
    assert manager._use_incluster_for_context("remote-cluster-1") is False
    assert manager._use_incluster_for_context("other-context") is False


def test_use_incluster_returns_true_when_context_matches_control_plane_node(
    mock_get_local_cluster_context,
):
    """When context matches control-plane node name (auto-detected), use in-cluster."""
    mock_get_local_cluster_context.return_value = "control-01"
    manager = KubernetesMultiClusterClientManager()
    assert manager._use_incluster_for_context("control-01") is True
    assert manager._use_incluster_for_context("control-02") is False
