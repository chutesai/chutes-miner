from unittest.mock import AsyncMock, create_autospec, patch

import pytest

from chutes_common.redis import MonitoringRedisClient


def _create_redis_mock():
    mock_instance = create_autospec(
        MonitoringRedisClient,
        instance=True,
        spec_set=True,
    )
    return mock_instance

@pytest.fixture(scope='module')
def mock_redis_client_instance():
    patches = [
        patch('chutes_monitor.api.main.MonitoringRedisClient'),
        patch('chutes_monitor.api.cluster.router.MonitoringRedisClient'),
        patch('chutes_monitor.cluster_monitor.MonitoringRedisClient'),
    ]
    
    mock_instance = _create_redis_mock()
    _patches = []
    
    for p in patches:
        mock_class = p.start()
        mock_class.return_value = mock_instance
        _patches.append(p)
    
    yield mock_instance
    
    # Clean up
    for p in _patches:
        p.stop()

@pytest.fixture(scope='function')
def mock_redis_client(mock_redis_client_instance):
    mock_redis_client_instance.reset_mock(side_effect=True)
    for attr_name in dir(mock_redis_client_instance):
        attr = getattr(mock_redis_client_instance, attr_name)
        if isinstance(attr, AsyncMock):
            attr.reset_mock(side_effect=True)
    yield mock_redis_client_instance