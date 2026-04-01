"""
Tests for TEE ConfigMap filtering.

Verifies that:
- Code ConfigMaps are never created for TEE chutes
- Code ConfigMaps are never deployed/deleted/synced to TEE clusters
- Non-TEE chute + non-TEE cluster behavior is unchanged
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from kubernetes.client import V1ConfigMap, V1ObjectMeta

from chutes_miner.api.k8s.operator import (
    K8sOperator,
    MultiClusterK8sOperator,
    ConfigMapWorker,
    _is_tee_cluster,
    _tee_cluster_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_tee_cache():
    _tee_cluster_cache.clear()
    yield
    _tee_cluster_cache.clear()


@pytest.fixture(autouse=True, scope="function")
def mock_multicluster_k8s_operator():
    original_new = K8sOperator.__new__

    def mock_new(cls, *args, **kwargs):
        return super(K8sOperator, cls).__new__(MultiClusterK8sOperator)

    K8sOperator.__new__ = mock_new
    K8sOperator._instance = None
    yield
    K8sOperator.__new__ = original_new
    K8sOperator._instance = None


@pytest.fixture(autouse=True)
def multicluster_setup(mock_k8s_client_manager, mock_redis_client, mock_watch):
    pass


def _make_operator(mock_redis_client, mock_k8s_client_manager):
    with patch("chutes_miner.api.k8s.operator.asyncio.create_task"), \
         patch.object(MultiClusterK8sOperator, "_watch_clusters", new=lambda self: None), \
         patch.object(MultiClusterK8sOperator, "_watch_cluster_connections", new=lambda self: None):
        return MultiClusterK8sOperator()


def _make_chute(tee=False):
    chute = MagicMock()
    chute.chute_id = "test-chute-id"
    chute.version = "1.0.0"
    chute.filename = "app.py"
    chute.code = "print('hello')"
    chute.tee = tee
    return chute


def _make_config_map(name="chute-code-abc"):
    return V1ConfigMap(
        metadata=V1ObjectMeta(
            name=name,
            labels={"chutes/chute-id": "test", "chutes/version": "1.0.0", "chutes/code": "true"},
        ),
        data={"app.py": "print('hello')"},
    )


def _make_worker(mock_redis_client, mock_k8s_client_manager):
    worker = ConfigMapWorker.__new__(ConfigMapWorker)
    worker._redis = mock_redis_client
    worker._manager = mock_k8s_client_manager
    worker._verify_node_health = MagicMock()
    worker._get_request_timeout = MagicMock(return_value=(5, 60))
    worker._build_code_config_map = MagicMock(side_effect=lambda c: _make_config_map())
    return worker


def _patch_tee_lookup(**kwargs):
    return patch("chutes_miner.api.k8s.operator._is_tee_cluster", **kwargs)


def _mock_sync_session(is_tee_value):
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar_one_or_none.return_value = is_tee_value
    ctx = patch("chutes_miner.api.k8s.operator.get_sync_session")
    mock_get = ctx.start()
    mock_get.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_get.return_value.__exit__ = MagicMock(return_value=False)
    return ctx, mock_session


# ---------------------------------------------------------------------------
# _is_tee_cluster – cache and DB lookup
# ---------------------------------------------------------------------------

def test_is_tee_cluster_returns_true_for_tee_server():
    ctx, _ = _mock_sync_session(True)
    try:
        assert _is_tee_cluster("tee-node") is True
    finally:
        ctx.stop()


def test_is_tee_cluster_returns_false_for_non_tee_server():
    ctx, _ = _mock_sync_session(False)
    try:
        assert _is_tee_cluster("gpu-node") is False
    finally:
        ctx.stop()


def test_is_tee_cluster_returns_false_for_unknown_server():
    ctx, _ = _mock_sync_session(None)
    try:
        assert _is_tee_cluster("no-such-node") is False
    finally:
        ctx.stop()


def test_is_tee_cluster_uses_cache_on_second_call():
    ctx, mock_session = _mock_sync_session(True)
    try:
        assert _is_tee_cluster("tee-node") is True
        assert _is_tee_cluster("tee-node") is True
        mock_session.execute.assert_called_once()
    finally:
        ctx.stop()


def test_is_tee_cluster_cache_expires_after_ttl():
    expired = datetime.now(timezone.utc) - timedelta(seconds=1)
    _tee_cluster_cache["old-node"] = (True, expired)

    ctx, mock_session = _mock_sync_session(False)
    try:
        assert _is_tee_cluster("old-node") is False
        mock_session.execute.assert_called_once()
    finally:
        ctx.stop()


# ---------------------------------------------------------------------------
# create_code_config_map – chute-level gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_code_config_map_skips_tee_chute(mock_redis_client, mock_k8s_core_client):
    chute = _make_chute(tee=True)
    await K8sOperator().create_code_config_map(chute)
    mock_k8s_core_client.create_namespaced_config_map.assert_not_called()


@pytest.mark.asyncio
async def test_create_code_config_map_skips_tee_chute_with_force(mock_redis_client, mock_k8s_core_client):
    chute = _make_chute(tee=True)
    await K8sOperator().create_code_config_map(chute, force=True)
    mock_k8s_core_client.create_namespaced_config_map.assert_not_called()


@pytest.mark.asyncio
async def test_create_code_config_map_builds_cm_for_non_tee_chute(mock_redis_client, mock_k8s_core_client):
    chute = _make_chute(tee=False)
    operator = K8sOperator()
    cm = operator._build_code_config_map(chute)
    assert cm.data["app.py"] == "print('hello')"
    assert "chute-code-" in cm.metadata.name


# ---------------------------------------------------------------------------
# MultiClusterK8sOperator – deploy_config_map_to_cluster
# ---------------------------------------------------------------------------

def test_multi_deploy_skips_tee_cluster(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    mock_redis_client.get_all_cluster_names.return_value = ["tee-node", "gpu-node"]

    with _patch_tee_lookup(side_effect=lambda c: c == "tee-node"):
        operator = _make_operator(mock_redis_client, mock_k8s_client_manager)
        operator._deploy_config_map_to_all_clusters(_make_config_map())

    assert mock_k8s_core_client.create_namespaced_config_map.call_count == 1


def test_multi_deploy_sends_to_all_non_tee_clusters(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    mock_redis_client.get_all_cluster_names.return_value = ["gpu-1", "gpu-2", "gpu-3"]

    with _patch_tee_lookup(return_value=False):
        operator = _make_operator(mock_redis_client, mock_k8s_client_manager)
        operator._deploy_config_map_to_all_clusters(_make_config_map())

    assert mock_k8s_core_client.create_namespaced_config_map.call_count == 3


def test_multi_deploy_sends_to_no_clusters_when_all_tee(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    mock_redis_client.get_all_cluster_names.return_value = ["tee-1", "tee-2"]

    with _patch_tee_lookup(return_value=True):
        operator = _make_operator(mock_redis_client, mock_k8s_client_manager)
        operator._deploy_config_map_to_all_clusters(_make_config_map())

    mock_k8s_core_client.create_namespaced_config_map.assert_not_called()


# ---------------------------------------------------------------------------
# MultiClusterK8sOperator – delete_config_map
# ---------------------------------------------------------------------------

def test_multi_delete_skips_tee_cluster(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    mock_redis_client.get_all_cluster_names.return_value = ["tee-node", "gpu-node"]

    with _patch_tee_lookup(side_effect=lambda c: c == "tee-node"):
        operator = _make_operator(mock_redis_client, mock_k8s_client_manager)
        operator.delete_config_map("chute-code-abc")

    assert mock_k8s_core_client.delete_namespaced_config_map.call_count == 1


def test_multi_delete_sends_to_all_non_tee_clusters(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    mock_redis_client.get_all_cluster_names.return_value = ["gpu-1", "gpu-2"]

    with _patch_tee_lookup(return_value=False):
        operator = _make_operator(mock_redis_client, mock_k8s_client_manager)
        operator.delete_config_map("chute-code-abc")

    assert mock_k8s_core_client.delete_namespaced_config_map.call_count == 2


# ---------------------------------------------------------------------------
# ConfigMapWorker – _deploy_config_map_to_cluster
# ---------------------------------------------------------------------------

def test_worker_deploy_skips_tee_cluster(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    worker = _make_worker(mock_redis_client, mock_k8s_client_manager)

    with _patch_tee_lookup(return_value=True):
        success, reason = worker._deploy_config_map_to_cluster(
            cluster="tee-node", config_map=_make_config_map(), namespace="chutes",
        )

    assert success is True
    assert reason is None
    mock_k8s_core_client.create_namespaced_config_map.assert_not_called()


def test_worker_deploy_proceeds_for_non_tee_cluster(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    worker = _make_worker(mock_redis_client, mock_k8s_client_manager)

    with _patch_tee_lookup(return_value=False):
        worker._deploy_config_map_to_cluster(
            cluster="gpu-node", config_map=_make_config_map(), namespace="chutes",
        )

    mock_k8s_core_client.create_namespaced_config_map.assert_called_once()


# ---------------------------------------------------------------------------
# ConfigMapWorker – _delete_config_map_from_cluster
# ---------------------------------------------------------------------------

def test_worker_delete_skips_tee_cluster(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    worker = _make_worker(mock_redis_client, mock_k8s_client_manager)

    with _patch_tee_lookup(return_value=True):
        worker._delete_config_map_from_cluster(
            cluster="tee-node", name="chute-code-abc", namespace="chutes",
        )

    mock_k8s_core_client.delete_namespaced_config_map.assert_not_called()


def test_worker_delete_proceeds_for_non_tee_cluster(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    worker = _make_worker(mock_redis_client, mock_k8s_client_manager)

    with _patch_tee_lookup(return_value=False):
        worker._delete_config_map_from_cluster(
            cluster="gpu-node", name="chute-code-abc", namespace="chutes",
        )

    mock_k8s_core_client.delete_namespaced_config_map.assert_called_once()


# ---------------------------------------------------------------------------
# ConfigMapWorker – _sync_cluster_configmaps
# ---------------------------------------------------------------------------

def test_worker_sync_skips_tee_cluster(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    worker = _make_worker(mock_redis_client, mock_k8s_client_manager)

    with _patch_tee_lookup(return_value=True):
        worker._sync_cluster_configmaps("tee-node")

    mock_k8s_core_client.list_namespaced_config_map.assert_not_called()
    mock_k8s_core_client.create_namespaced_config_map.assert_not_called()
    mock_k8s_core_client.delete_namespaced_config_map.assert_not_called()


# ---------------------------------------------------------------------------
# ConfigMapWorker – _deploy_config_map_to_all_clusters (mixed)
# ---------------------------------------------------------------------------

def test_worker_deploy_all_skips_tee_clusters_only(mock_redis_client, mock_k8s_core_client, mock_k8s_client_manager):
    worker = _make_worker(mock_redis_client, mock_k8s_client_manager)
    mock_redis_client.get_all_cluster_names.return_value = ["tee-1", "gpu-1", "tee-2", "gpu-2"]

    with _patch_tee_lookup(side_effect=lambda c: c.startswith("tee")):
        failures = worker._deploy_config_map_to_all_clusters(
            config_map=_make_config_map(), namespace="chutes",
        )

    assert mock_k8s_core_client.create_namespaced_config_map.call_count == 2
    assert len(failures) == 0
