"""Unit tests for TEE verification strategy."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from kubernetes.client import V1Node, V1ObjectMeta

from chutes_common.schemas.server import ServerArgs
from chutes_miner.api.exceptions import TEEBootstrapFailure
from chutes_miner.common.verification import TEEVerificationStrategy


@pytest.fixture
def mock_node():
    """Create a mock V1Node object for TEE."""
    node = Mock(spec=V1Node)
    node.metadata = Mock(spec=V1ObjectMeta)
    node.metadata.labels = {
        "nvidia.com/gpu.count": "2",
        "gpu-short-ref": "RTX4090",
        "chutes/external-ip": "192.168.1.100",
        "chutes/tee": "true",
    }
    node.metadata.name = "test-node"
    node.metadata.uid = "test-node-uid"
    return node


@pytest.fixture
def mock_server():
    """Create a mock Server object."""
    server = Mock()
    server.server_id = "test-node-uid"
    server.name = "test-node"
    server.cpu_per_gpu = 8
    server.memory_per_gpu = 32
    server.ip_address = "192.168.1.100"
    server.verification_port = 30443
    return server


@pytest.fixture
def mock_devices():
    """Create mock device data for TEE."""
    return [
        {
            "uuid": "gpu_1",
            "name": "NVIDIA GeForce RTX 4090",
            "memory": 24 * 1024,
        },
        {
            "uuid": "gpu_2",
            "name": "NVIDIA GeForce RTX 4090",
            "memory": 24 * 1024,
        },
    ]


@pytest.fixture(autouse=True)
def mock_validator_lookup():
    """Mock validator_by_hotkey for TEE tests."""
    def _build_validator(hotkey: str):
        validator_obj = Mock()
        validator_obj.hotkey = hotkey
        validator_obj.api = f"http://{hotkey}.example.com"
        return validator_obj

    with patch(
        "chutes_miner.common.verification.validator_by_hotkey",
        side_effect=_build_validator,
    ):
        yield


async def gather_gpu_info_tee(
    *,
    server_id: str,
    validator: str,
    node_object: V1Node,
    server,
):
    """Helper to test TEE gather_gpu_info. Returns strategy.gpus after gather."""
    server_args = ServerArgs(
        name="test-server",
        validator=validator,
        hourly_cost=1.0,
        gpu_short_ref=node_object.metadata.labels.get("gpu-short-ref", "unknown"),
    )
    strategy = TEEVerificationStrategy(node_object, server_args, server)
    await strategy.gather_gpu_info()
    return strategy.gpus


@pytest.mark.asyncio
async def test_tee_successful_gpu_gathering(mock_node, mock_server, mock_devices):
    """Test successful GPU information gathering from attestation service."""
    mock_session = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_gpu_instances = [Mock() for _ in range(2)]
    for gpu in mock_gpu_instances:
        gpu.device_info = {"name": "RTX 4090", "uuid": "gpu_1"}

    with (
        patch.object(
            TEEVerificationStrategy, "_fetch_devices", AsyncMock(return_value=mock_devices)
        ) as mock_fetch,
        patch("chutes_miner.common.verification.get_session") as mock_get_session,
        patch("chutes_miner.common.verification.GPU") as mock_gpu_class,
    ):
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_gpu_class.side_effect = mock_gpu_instances

        gpus = await gather_gpu_info_tee(
            server_id="server-123",
            validator="validator-456",
            node_object=mock_node,
            server=mock_server,
        )

        mock_fetch.assert_called_once()
        assert len(gpus) == 2
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_tee_attestation_service_failure(mock_node, mock_server):
    """Test failure when attestation service health check fails."""
    with patch.object(
        TEEVerificationStrategy,
        "_verify_attestation_service",
        side_effect=TEEBootstrapFailure("Failed to verify attestion service"),
    ):
        strategy = TEEVerificationStrategy(
            mock_node,
            ServerArgs(name="test", validator="val", hourly_cost=1.0, gpu_short_ref="RTX4090"),
            mock_server,
        )

        with pytest.raises(TEEBootstrapFailure, match="Failed to verify attestion service"):
            await strategy.prepare_verification_environment()


@pytest.mark.asyncio
async def test_tee_device_fetch_failure(mock_node, mock_server):
    """Test failure when fetching devices from attestation service."""
    with patch.object(
        TEEVerificationStrategy,
        "_fetch_devices",
        AsyncMock(side_effect=Exception("Connection refused")),
    ):
        with pytest.raises(
            TEEBootstrapFailure,
            match="Failed to fetch devices from attestation service",
        ):
            await gather_gpu_info_tee(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                server=mock_server,
            )


@pytest.mark.asyncio
async def test_tee_gpu_count_mismatch(mock_node, mock_server):
    """Test failure when device count doesn't match expected GPU count."""
    mock_devices = [{"uuid": "gpu_1", "name": "RTX 4090"}]  # Only 1 when 2 expected

    with patch.object(
        TEEVerificationStrategy,
        "_fetch_devices",
        AsyncMock(return_value=mock_devices),
    ):
        with pytest.raises(
            TEEBootstrapFailure,
            match="Failed to fetch devices from attestation service",
        ):
            await gather_gpu_info_tee(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                server=mock_server,
            )


@pytest.mark.asyncio
async def test_tee_empty_devices_response(mock_node, mock_server):
    """Test failure when devices response is empty."""
    with patch.object(
        TEEVerificationStrategy,
        "_fetch_devices",
        AsyncMock(return_value=[]),
    ):
        with pytest.raises(
            TEEBootstrapFailure,
            match="Failed to fetch devices from attestation service",
        ):
            await gather_gpu_info_tee(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                server=mock_server,
            )


@pytest.mark.asyncio
async def test_tee_none_devices_response(mock_node, mock_server):
    """Test failure when devices response is None."""
    with patch.object(
        TEEVerificationStrategy,
        "_fetch_devices",
        AsyncMock(return_value=None),
    ):
        with pytest.raises(
            TEEBootstrapFailure,
            match="Failed to fetch devices from attestation service",
        ):
            await gather_gpu_info_tee(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                server=mock_server,
            )
