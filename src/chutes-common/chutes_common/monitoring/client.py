"""
Client for agent monitoring API (e.g. /monitor/start, /monitor/stop).
Shared by miner and monitor so request/response handling is consistent.
"""

import aiohttp

from chutes_common.auth import sign_request
from chutes_common.exceptions import AgentError
from chutes_common.monitoring.requests import StartMonitoringRequest


async def start_server_monitoring(
    agent_url: str,
    control_plane_url: str,
    timeout: int = 30,
) -> None:
    """
    POST signed start-monitoring request to the agent. The agent will register
    with the control plane and begin sending heartbeats.

    Args:
        agent_url: Base URL of the agent API (e.g. http://agent:8000).
        control_plane_url: URL to send to the agent as the control plane (monitor API).
        timeout: Request timeout in seconds.

    Raises:
        AgentError: If the agent returns a non-200 status (includes status_code and response text).
    """
    request = StartMonitoringRequest(control_plane_url=control_plane_url)
    payload = request.model_dump()
    headers, payload_string = sign_request(payload, purpose="monitoring", management=True)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        url = f"{agent_url.rstrip('/')}/monitor/start"
        async with session.post(
            url,
            data=payload_string,
            headers=headers,
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise AgentError(text, status_code=response.status)


async def stop_server_monitoring(
    agent_url: str,
    conn_timeout: int = 5,
    read_timeout: int = 30,
) -> None:
    """
    GET /monitor/stop on the agent to stop monitoring and remove the cluster from cache.

    Args:
        agent_url: Base URL of the agent API.
        conn_timeout: Connection timeout in seconds.
        read_timeout: Read timeout in seconds.

    Raises:
        AgentError: If the agent returns a non-200 status.
    """
    headers, _ = sign_request(purpose="monitoring", management=True)
    timeout = aiohttp.ClientTimeout(connect=conn_timeout, total=read_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        url = f"{agent_url.rstrip('/')}/monitor/stop"
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                raise AgentError(text, status_code=response.status)
