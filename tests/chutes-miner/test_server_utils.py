import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch
from kubernetes.client import (
    V1Node,
    V1Deployment,
    V1Service,
    V1ObjectMeta,
    V1JobCondition,
    V1ServiceSpec,
    V1ServicePort,
    V1DeploymentSpec,
    V1Job
)
from chutes_miner.api.server.util import gather_gpu_info, GPU, GraValBootstrapFailure


@pytest.fixture
def mock_node():
    """Create a mock V1Node object"""
    node = Mock(spec=V1Node)
    node.metadata = Mock(spec=V1ObjectMeta)
    node.metadata.labels = {
        "nvidia.com/gpu.count": "2",
        "gpu-short-ref": "RTX4090",
        "chutes/external-ip": "192.168.1.100",
    }
    return node


@pytest.fixture
def mock_job():
    """Create a mock V1Job object"""
    deployment = Mock(spec=V1Job)
    deployment.metadata = Mock(spec=V1ObjectMeta)
    deployment.metadata.name = "test-deployment"
    deployment.metadata.namespace = "test-namespace"
    deployment.spec = Mock(spec=V1DeploymentSpec)
    deployment.spec.replicas = 1
    return deployment

@pytest.fixture
def mock_deployment():
    """Create a mock V1Deployment object"""
    deployment = Mock(spec=V1Deployment)
    deployment.metadata = Mock(spec=V1ObjectMeta)
    deployment.metadata.name = "test-deployment"
    deployment.metadata.namespace = "test-namespace"
    deployment.spec = Mock(spec=V1DeploymentSpec)
    deployment.spec.replicas = 1
    return deployment


@pytest.fixture
def mock_service():
    """Create a mock V1Service object"""
    service = Mock(spec=V1Service)
    service.spec = Mock(spec=V1ServiceSpec)
    port = Mock(spec=V1ServicePort)
    port.node_port = 30080
    service.spec.ports = [port]
    return service


@pytest.fixture
def mock_devices():
    """Create mock device data"""
    return [
        {
            "uuid": "GPU-12345678-1234-1234-1234-123456789012",
            "name": "NVIDIA GeForce RTX 4090",
            "memory": "24GB",
        },
        {
            "uuid": "GPU-87654321-4321-4321-4321-210987654321",
            "name": "NVIDIA GeForce RTX 4090",
            "memory": "24GB",
        },
    ]


@pytest.mark.asyncio
async def test_successful_gpu_gathering(
    mock_node, mock_job, mock_service, mock_devices
):
    """Test successful GPU information gathering"""
    # Mock the deployment watch stream
    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job

    # Mock database session
    mock_session = AsyncMock()
    mock_gpu_instances = [Mock(spec=GPU) for _ in range(2)]

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch(
            "chutes_miner.api.server.util._fetch_devices", return_value=mock_devices
        ) as mock_fetch,
        patch("chutes_miner.api.server.util.get_session") as mock_get_session,
        patch("chutes_miner.api.server.util.GPU") as mock_gpu_class,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        # Setup mocks
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_gpu_class.side_effect = mock_gpu_instances

        # Execute
        result = await gather_gpu_info(
            server_id="server-123",
            validator="validator-456",
            node_object=mock_node,
            graval_job=mock_job,
            graval_service=mock_service,
        )

        # Assertions
        assert len(result) == 2
        mock_fetch.assert_called_once_with("http://192.168.1.100:30080/devices")
        assert mock_session.add.call_count == 2
        mock_session.commit.assert_called_once()
        assert mock_session.refresh.call_count == 2


@pytest.mark.asyncio
async def test_missing_gpu_short_ref_label(mock_job, mock_service):
    """Test failure when gpu-short-ref label is missing"""
    node = Mock(spec=V1Node)
    node.metadata = Mock(spec=V1ObjectMeta)
    node.metadata.labels = {
        "nvidia.com/gpu.count": "2",
        "chutes/external-ip": "192.168.1.100",
        # Missing gpu-short-ref
    }

    with pytest.raises(
        GraValBootstrapFailure, match="Node does not have required gpu-short-ref label!"
    ):
        await gather_gpu_info(
            server_id="server-123",
            validator="validator-456",
            node_object=node,
            graval_job=mock_job,
            graval_service=mock_service,
        )


@pytest.mark.asyncio
async def test_job_failure_condition(mock_node, mock_job, mock_service):
    """Test handling of job failure conditions"""
    mock_failed_condition = Mock(spec=V1JobCondition)
    mock_failed_condition.type = "Failed"
    mock_failed_condition.status = "True"
    mock_failed_condition.message = "Pod crashed"

    mock_failed_job = Mock()
    mock_failed_job.status = Mock()
    mock_failed_job.status.conditions = [mock_failed_condition]
    mock_failed_job.status.phase = "Failed"
    mock_failed_job.status.message = "Pod crashed"

    mock_watch_event = Mock()
    mock_watch_event.object = mock_failed_job

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])

        with pytest.raises(GraValBootstrapFailure, match="Bootstrap pod failed: Pod crashed"):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_deployment_timeout(
    mock_node, mock_job, mock_service
):
    """Test deployment timeout handling"""
    mock_not_ready_job = Mock()
    mock_not_ready_job.status = Mock()
    mock_not_ready_job.status.ready_replicas = 0
    mock_not_ready_job.status.conditions = []
    mock_not_ready_job.spec = Mock()
    mock_not_ready_job.spec.replicas = 1

    mock_watch_event = Mock()
    mock_watch_event.object = mock_not_ready_job

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
        patch("time.time", side_effect=[0, 65]),
    ):  # Simulate timeout
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])

        with pytest.raises(
            GraValBootstrapFailure, match="Error waiting for graval bootstrap job: GraVal bootstrap job not ready after 65 seconds!"
        ):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_watch_stream_exception(mock_node, mock_job, mock_service):
    """Test handling of exceptions during deployment watching"""
    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.side_effect = Exception("Connection error")

        with pytest.raises(
            GraValBootstrapFailure,
            match="Error waiting for graval bootstrap job: Connection error",
        ):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_device_fetch_failure(mock_node, mock_job, mock_service):
    """Test failure when fetching device information"""
    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch(
            "chutes_miner.api.server.util._fetch_devices",
            side_effect=Exception("Connection failed"),
        ),
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])

        with pytest.raises(
            GraValBootstrapFailure, match="Failed to fetch devices from GraVal bootstrap"
        ):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_gpu_count_mismatch(mock_node, mock_job, mock_service):
    """Test failure when device count doesn't match expected GPU count"""
    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job

    # Return only 1 device when 2 are expected
    mock_devices = [{"uuid": "GPU-12345", "name": "RTX 4090"}]

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util._fetch_devices", return_value=mock_devices),
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])

        with pytest.raises(
            GraValBootstrapFailure, match="Failed to fetch devices from GraVal bootstrap"
        ):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_no_node_port_in_service(mock_node, mock_job):
    """Test handling when service has no node port"""
    service = Mock(spec=V1Service)
    service.spec = Mock(spec=V1ServiceSpec)
    port_without_nodeport = Mock(spec=V1ServicePort)
    port_without_nodeport.node_port = None
    service.spec.ports = [port_without_nodeport]

    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch(
            "chutes_miner.api.server.util._fetch_devices",
            side_effect=Exception("Connection failed"),
        ),
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.watch_deployments.return_value = iter([mock_watch_event])

        with pytest.raises(GraValBootstrapFailure):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=service,
            )


@pytest.mark.asyncio
async def test_default_namespace_fallback(
    mock_node, mock_service, mock_devices
):
    """Test fallback to 'chutes' namespace when deployment namespace is None"""
    job = Mock(spec=V1Deployment)
    job.metadata = Mock(spec=V1ObjectMeta)
    job.metadata.name = "test-deployment"
    job.metadata.namespace = None  # Test fallback
    job.spec = Mock(spec=V1DeploymentSpec)
    job.spec.replicas = 1
    

    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job
    mock_session = AsyncMock()

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util._fetch_devices", return_value=mock_devices),
        patch("chutes_miner.api.server.util.get_session") as mock_get_session,
        patch("chutes_miner.api.server.util.GPU") as mock_gpu_class,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_gpu_class.return_value = Mock(spec=GPU)

        await gather_gpu_info(
            server_id="server-123",
            validator="validator-456",
            node_object=mock_node,
            graval_job=job,
            graval_service=mock_service,
        )

        # Verify that the watch was called with 'chutes' namespace
        mock_operator.return_value.watch_pods.assert_called_once_with(
            namespace="chutes", label_selector=ANY, timeout=ANY
        )


@pytest.mark.asyncio
async def test_zero_gpu_count(mock_job, mock_service):
    """Test handling of node with zero GPUs"""
    node = Mock(spec=V1Node)
    node.metadata = Mock(spec=V1ObjectMeta)
    node.metadata.labels = {
        "nvidia.com/gpu.count": "0",
        "gpu-short-ref": "RTX4090",
        "chutes/external-ip": "192.168.1.100",
    }

    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job
    mock_session = AsyncMock()

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util._fetch_devices", return_value=[]),
        patch("chutes_miner.api.server.util.get_session") as mock_get_session,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])
        mock_get_session.return_value.__aenter__.return_value = mock_session

        with pytest.raises(
            GraValBootstrapFailure, match="Failed to fetch devices from GraVal bootstrap"
        ):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_database_commit_failure(mock_node, mock_job, mock_service, mock_devices):
    """Test handling of database commit failures"""
    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job
    mock_session = AsyncMock()
    mock_session.commit.side_effect = Exception("Database error")

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util._fetch_devices", return_value=mock_devices),
        patch("chutes_miner.api.server.util.get_session") as mock_get_session,
        patch("chutes_miner.api.server.util.GPU") as mock_gpu_class,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_gpu_class.return_value = Mock(spec=GPU)

        with pytest.raises(Exception, match="Database error"):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_missing_external_ip_label(mock_job, mock_service):
    """Test handling when node is missing external IP label"""
    node = Mock(spec=V1Node)
    node.metadata = Mock(spec=V1ObjectMeta)
    node.metadata.labels = {
        "nvidia.com/gpu.count": "2",
        "gpu-short-ref": "RTX4090",
        # Missing chutes/external-ip
    }

    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]
    

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch(
            "chutes_miner.api.server.util._fetch_devices",
            side_effect=Exception("Connection failed"),
        ),
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.watch_pods.return_value = iter([mock_watch_event])

        with pytest.raises(GraValBootstrapFailure):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_empty_devices_response(mock_node, mock_job, mock_service):
    """Test handling when devices response is empty but GPUs are expected"""
    mock_ready_job = Mock()
    mock_ready_job.status = Mock()
    mock_ready_job.status.ready_replicas = 1
    mock_ready_job.status.conditions = []
    mock_ready_job.spec = Mock()
    mock_ready_job.spec.replicas = 1
    mock_ready_job.status.phase = "Running"
    mock_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = Mock()
    mock_watch_event.object = mock_ready_job

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util._fetch_devices", return_value=None),
        patch("chutes_miner.api.server.util.settings") as mock_settings,
    ):
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.return_value.watch_pods.return_value = iter([mock_watch_event])

        with pytest.raises(
            GraValBootstrapFailure, match="Failed to fetch devices from GraVal bootstrap"
        ):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )


@pytest.mark.asyncio
async def test_deployment_never_becomes_ready(mock_node, mock_job, mock_service):
    """Test when deployment watch completes but deployment is never ready"""
    mock_not_ready_job = Mock()
    mock_not_ready_job.status = Mock()
    mock_not_ready_job.status.ready_replicas = 0
    mock_not_ready_job.status.conditions = []
    mock_not_ready_job.spec = Mock()
    mock_not_ready_job.spec.replicas = 1
    mock_not_ready_job.status.phase = "Pending"
    mock_not_ready_job.status.container_statuses = [MagicMock(ready=True)]

    mock_watch_event = {"object": mock_not_ready_job}

    with (
        patch("chutes_miner.api.server.util.K8sOperator") as mock_operator,
        patch("chutes_miner.api.server.util.settings") as mock_settings,
        patch("time.time", return_value=0),
    ):  # No timeout, just stream ends
        mock_settings.graval_bootstrap_timeout = 60
        mock_operator.watch_pods.return_value = iter([mock_watch_event])

        with pytest.raises(
            GraValBootstrapFailure, match="GraVal bootstrap job never reached ready state"
        ):
            await gather_gpu_info(
                server_id="server-123",
                validator="validator-456",
                node_object=mock_node,
                graval_job=mock_job,
                graval_service=mock_service,
            )
