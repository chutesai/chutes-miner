from unittest.mock import AsyncMock, Mock, patch

from chutes_common.k8s import WatchEvent, WatchEventType

import pytest


from fixtures.bootstrap_fixtures import * # noqa

@pytest.fixture(autouse=True)
def mock_strategy_factory(mock_node):
    mock_node.metadata.labels["chutes/tee"] = "true"
    yield mock_node

@pytest.fixture(autouse=True)
def mock_fetch_devices(mock_gpus):

    _mock = AsyncMock()
    with patch("chutes_miner.api.server.verification.TEEVerificationStrategy._fetch_devices", _mock):
        _mock.return_value = [
        {
            "uuid": gpu.gpu_id,
            **gpu.device_info
        } for gpu in mock_gpus
    ]
        yield _mock

@pytest.fixture(autouse=True)
def mock_server_response(mock_aiohttp_response):
    mock_aiohttp_response.status = 201
    yield mock_aiohttp_response

@pytest.mark.asyncio
async def test_bootstrap_server_success_without_kubeconfig(
    mock_node, mock_server_args, mock_track_server, mock_aiohttp_response
):
    """Test successful server bootstrapping without kubeconfig"""
    # Mock aiohttp session for advertise nodes
    from chutes_miner.api.server.util import bootstrap_server
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args, None)
    )
    
    # Assertions
    assert len(messages) > 0
    assert any("attempting to add node" in str(msg) for msg in messages)
    assert any("completed server bootstrapping" in str(msg) for msg in messages)
    
    # Verify key method calls
    mock_track_server.assert_called_once()


@pytest.mark.asyncio
async def test_bootstrap_server_success_with_kubeconfig(
    mock_node, mock_server_args, mock_kubeconfig, mock_multicluster_kubeconfig
):
    """Test successful server bootstrapping with kubeconfig"""
    
    # Execute test
    from chutes_miner.api.server.util import bootstrap_server
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args, mock_kubeconfig)
    )
    
    # Assertions
    assert len(messages) > 0
    mock_multicluster_kubeconfig.return_value.add_config.assert_called_once_with(mock_kubeconfig)


@pytest.mark.asyncio
async def test_bootstrap_server_success_with_agent_api(
    mock_node, mock_server_args_with_agent, mock_dependencies,
    mock_start_server_monitoring, mock_stop_server_monitoring
):
    """Test successful server bootstrapping with agent API monitoring"""
    # Execute test
    from chutes_miner.api.server.util import bootstrap_server
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args_with_agent, None)
    )
    
    # Assertions
    assert len(messages) > 0
    mock_start_server_monitoring.assert_called_once_with("http://agent-api.com")
    mock_stop_server_monitoring.assert_not_called()


@pytest.mark.asyncio
async def test_bootstrap_server_fetch_devices_failure(
    mock_node, mock_server_args, mock_aiohttp_session, 
    set_mock_db_session_result, mock_server, mock_gpus, 
    mock_fetch_devices
):
    """Test bootstrap failure due to GPU verification failure"""
    mock_fetch_devices.side_effect = Exception("Failure to fetch.")

    mock_server.gpus = mock_gpus
    set_mock_db_session_result([mock_server])

    # Execute test and expect exception
    from chutes_miner.api.server.util import bootstrap_server
    from chutes_miner.api.exceptions import TEEBootstrapFailure
    
    with pytest.raises(TEEBootstrapFailure):
        await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )

    # Verify server was cleaned up from validator
    assert mock_aiohttp_session.delete.call_count == 1  # One server


@pytest.mark.asyncio
async def test_bootstrap_server_advertise_server_failure(
    mock_node, mock_server_args, mock_server,
    mock_gpus, set_mock_db_session_result, mock_aiohttp_session
):
    """Test bootstrap failure during node advertisement"""
    # Setup mocks
    mock_server.gpus = mock_gpus
    set_mock_db_session_result([mock_server])

    with patch("chutes_miner.api.server.verification.TEEVerificationStrategy._advertise_server") as mock_advertise:
        mock_advertise.side_effect = Exception("Advertisement failed")
    
        # Execute test and expect exception
        from chutes_miner.api.server.util import bootstrap_server
        
        with pytest.raises(Exception, match="Advertisement failed"):
            await collect_sse_messages(
                bootstrap_server(mock_node, mock_server_args, None)
            )

    # Verify server was cleaned up from validator
    assert mock_aiohttp_session.delete.call_count == 1  # One server


@pytest.mark.asyncio
async def test_bootstrap_server_track_server_failure(
    mock_node, mock_server_args, mock_track_server
):
    """Test bootstrap failure during server tracking"""
    # Setup mocks
    mock_track_server.side_effect = Exception("Tracking failed")
    
    # Execute test and expect exception
    from chutes_miner.api.server.util import bootstrap_server
    
    with pytest.raises(Exception, match="Tracking failed"):
        await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )


# @pytest.mark.asyncio 
# async def test_bootstrap_server_verification_timeout_simulation(
#     mock_node, mock_server_args, mock_check_verification_task_status
# ):
#     """Test verification status checking with multiple polls before success"""
    
#     # Simulate verification polling - return None twice, then True
#     mock_check_verification_task_status.side_effect = [None, None, True]
    
#     # Execute test
#     from chutes_miner.api.server.util import bootstrap_server
#     messages = await collect_sse_messages(
#         bootstrap_server(mock_node, mock_server_args, None)
#     )
    
#     # Verify verification was checked multiple times
#     mock_check_verification_task_status.call_count == 3
    
#     # Verify we got waiting messages
#     waiting_messages = [msg for msg in messages if "waiting for validator" in str(msg)]
#     assert len(waiting_messages) >= 2


@pytest.mark.asyncio
async def test_bootstrap_server_timing_measurement(
    mock_node, mock_server_args
):
    """Test that bootstrap timing is properly measured and reported"""
    
    # Mock time to control timing
    start_time = 1000.0
    end_time = 1045.5  # 45.5 seconds
    
    with patch('chutes_miner.api.server.util.time') as mock_time:
        mock_time.time.side_effect=[start_time, end_time]
        from chutes_miner.api.server.util import bootstrap_server
        messages = await collect_sse_messages(
            bootstrap_server(mock_node, mock_server_args, None)
        )
    
    # Verify timing is reported in final message
    final_messages = [msg for msg in messages if "completed server bootstrapping" in str(msg)]
    assert len(final_messages) == 1
    assert "45.5 seconds" in str(final_messages[0])


# Integration-style test that doesn't mock everything
@pytest.mark.asyncio
async def test_bootstrap_server_sse_message_flow(mock_node, mock_server_args):
    """Test that SSE messages are yielded in the expected order"""
    from chutes_miner.api.server.util import bootstrap_server
    
    messages = await collect_sse_messages(
        bootstrap_server(mock_node, mock_server_args, None)
    )
    
    # Verify message flow
    message_strs = [str(msg) for msg in messages]
    
    # Check that messages appear in expected order
    assert any("attempting to add node" in msg for msg in message_strs)
    assert any("now tracked in database" in msg for msg in message_strs)
    assert any("Verified attestation service" in msg for msg in message_strs)
    assert any("discovered" in msg and "GPUs" in msg for msg in message_strs)
    assert any("Verifying server" in msg for msg in message_strs)
    assert any("successfully verifeid server" in msg for msg in message_strs)
    assert any("completed server bootstrapping" in msg for msg in message_strs)