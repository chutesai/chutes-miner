from re import L
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.fixture
def mock_aiohttp_response():
    mock_response = MagicMock()
    mock_response.status = 202
    mock_response.text = AsyncMock(return_value = "")
    mock_response.json = AsyncMock(return_value = {
        "nodes": [
            {
                "seed": "abcd1234"
            },
            {
                "seed": "abcd1234"
            }
        ],
        "task_id": "verification-task"
    })

    return mock_response

@pytest.fixture
def mock_aiohttp_session(mock_aiohttp_response):
    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_aiohttp_response)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_aiohttp_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_session.put.return_value.__aenter__ = AsyncMock(return_value=mock_aiohttp_response)
    mock_session.put.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_session.delete.return_value.__aenter__ = AsyncMock(return_value=mock_aiohttp_response)
    mock_session.delete.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_session.patch.return_value.__aenter__ = AsyncMock(return_value=mock_aiohttp_response)
    mock_session.patch.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_session

@pytest.fixture(autouse=True)
def mock_aiohttp_client_session(mock_aiohttp_session):
    """Mock aiohttp session"""
    with patch("aiohttp.ClientSession") as mock_client_session:
        mock_client_session.return_value = mock_aiohttp_session            
        mock_client_session.return_value.__aenter__.return_value = mock_aiohttp_session
        mock_client_session.return_value.__aexit__.return_value = None

        yield mock_client_session