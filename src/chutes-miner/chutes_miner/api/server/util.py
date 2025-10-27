"""
Server utility functions.
"""

import asyncio
import json
import time
import math
import aiohttp
import traceback
from chutes_common.monitoring.requests import StartMonitoringRequest
from chutes_common.redis import MonitoringRedisClient
from chutes_miner.api.k8s.config import KubeConfig, MultiClusterKubeConfig
from chutes_miner.api.server.verification import VerificationStrategy
from loguru import logger
from kubernetes.client import V1Node
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from typing import Optional, Tuple, Dict
from chutes_common.auth import sign_request
from chutes_miner.api.config import settings
from chutes_miner.api.k8s.operator import K8sOperator
from chutes_miner.api.util import sse_message
from chutes_miner.api.database import get_session
from chutes_common.schemas.server import Server, ServerArgs
from chutes_common.schemas.gpu import GPU
from chutes_miner.api.exceptions import DuplicateServer
import yaml


async def get_server_kubeconfig(agent_url: str):
    async with aiohttp.ClientSession() as session:
        headers, _ = sign_request(purpose="registration", management=True)
        async with session.get(
            f"{agent_url}/config/kubeconfig",
            headers=headers,
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to retrieve kubeconfig from {agent_url}.")

            try:
                data = await response.json()
                return KubeConfig.from_dict(yaml.safe_load(data["kubeconfig"]))
            except Exception as err:
                raise RuntimeError(f"Failed to retrieve kubeconfig from {agent_url}:\n{err}")


async def start_server_monitoring(agent_url: str):
    request = StartMonitoringRequest(control_plane_url=settings.monitor_api)
    payload = request.model_dump()
    async with aiohttp.ClientSession() as session:
        headers, payload_string = sign_request(payload, purpose="monitoring", management=True)
        async with session.post(
            f"{agent_url}/monitor/start",
            data=payload_string,
            headers=headers,
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to start monitoring for cluster: {await response.text()}")


async def stop_server_monitoring(agent_url: str):
    async with aiohttp.ClientSession(conn_timeout=5, read_timeout=30) as session:
        headers, _ = sign_request(purpose="monitoring", management=True)
        async with session.get(
            f"{agent_url}/monitor/stop",
            headers=headers,
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to stop monitoring for cluster: {await response.text()}")


async def clear_server_cache(cluster_name):
    _redis = MonitoringRedisClient()
    await _redis.clear_cluster(cluster_name)
    _redis.close()


async def track_server(
    validator: str,
    hourly_cost: float,
    node_object: V1Node,
    add_labels: Dict[str, str] = None,
    agent_api: Optional[str] = None,
    kubeconfig: Optional[KubeConfig] = None,
) -> Tuple[V1Node, Server]:
    """
    Track a new kubernetes (worker/GPU) node in our inventory.
    """
    if not node_object.metadata or not node_object.metadata.name:
        raise ValueError("Node object must have metadata and name")

    # Make sure the labels (in kubernetes) are up-to-date.
    current_labels = node_object.metadata.labels or {}
    labels_to_add = {}
    for key, value in (add_labels or {}).items():
        if key not in current_labels or current_labels[key] != value:
            labels_to_add[key] = value
    if labels_to_add:
        current_labels.update(labels_to_add)
        body = {"metadata": {"labels": current_labels}}
        node_object = K8sOperator().patch_node(name=node_object.metadata.name, body=body)
    labels = current_labels

    # Extract node information from kubernetes meta.
    name = node_object.metadata.name
    server_id = node_object.metadata.uid
    ip_address = node_object.metadata.labels.get("chutes/external-ip")

    # Determine node status.
    status = "Unknown"
    if node_object.status and node_object.status.conditions:
        for condition in node_object.status.conditions:
            if condition.type == "Ready":
                status = "Ready" if condition.status == "True" else "NotReady"
                break
    if status != "Ready":
        raise ValueError(f"Node is not yet ready [{status=}]")

    # Calculate CPU/RAM per GPU for allocation purposes.
    gpu_count = int(node_object.status.capacity["nvidia.com/gpu"])
    gpu_mem_mb = int(node_object.metadata.labels.get("nvidia.com/gpu.memory", "32"))
    gpu_mem_gb = int(gpu_mem_mb / 1024)
    cpu_count = (
        int(node_object.status.capacity["cpu"]) - 2
    )  # leave 2 CPUs for incidentals, daemon sets, etc.
    cpu_per_gpu = 1 if cpu_count <= gpu_count else min(4, math.floor(cpu_count / gpu_count))
    raw_mem = node_object.status.capacity["memory"]
    if raw_mem.endswith("Ki"):
        total_memory_gb = int(int(raw_mem.replace("Ki", "")) / 1024 / 1024) - 6
    elif raw_mem.endswith("Mi"):
        total_memory_gb = int(int(raw_mem.replace("Mi", "")) / 1024) - 6
    elif raw_mem.endswith("Gi"):
        total_memory_gb = int(raw_mem.replace("Gi", "")) - 6
    memory_per_gpu = (
        1
        if total_memory_gb <= gpu_count
        else min(gpu_mem_gb, math.floor(total_memory_gb * 0.8 / gpu_count))
    )

    _kubeconfig = json.dumps(kubeconfig.to_dict()) if kubeconfig else None

    # Track the server in our inventory.
    async with get_session() as session:
        server = Server(
            server_id=node_object.metadata.uid,
            validator=validator,
            name=name,
            ip_address=ip_address,
            status=status,
            labels=labels,
            gpu_count=gpu_count,
            cpu_per_gpu=cpu_per_gpu,
            memory_per_gpu=memory_per_gpu,
            hourly_cost=hourly_cost,
            kubeconfig=_kubeconfig,
            agent_api=agent_api,
        )
        session.add(server)
        try:
            await session.commit()
        except IntegrityError as exc:
            if "UniqueViolationError" in str(exc):
                raise DuplicateServer(
                    f"Server {server_id=} {name=} {server_id=} already in database."
                )
            else:
                raise
        await session.refresh(server)

    return node_object, server


async def bootstrap_server(
    node_object: V1Node, server_args: ServerArgs, kubeconfig: Optional[KubeConfig]
):
    """
    Bootstrap a server from start to finish, yielding SSEs for miner to track status.
    """
    started_at = time.time()
    strategy = None

    async def _cleanup(delete_node=False):
        if delete_node and server_args.agent_api:
            # If adding a standalone cluster, need to stop monitoring
            await stop_server_monitoring(server_args.agent_api)
            logger.info(f"Stopped monitoring for {server_args.name}")

    yield sse_message(
        f"attempting to add node server_id={node_object.metadata.uid} to inventory...",
    )

    try:
        if kubeconfig:
            # Make sure this is available for deploying
            MultiClusterKubeConfig().add_config(kubeconfig)

        node, server = await track_server(
            server_args.validator,
            server_args.hourly_cost,
            node_object,
            add_labels={
                "gpu-short-ref": server_args.gpu_short_ref,
                "chutes/validator": server_args.validator,
                "chutes/worker": "true",
            },
            agent_api=server_args.agent_api,
            kubeconfig=kubeconfig,
        )

        # Great, now it's in our database, but we need to startup resources so the validator can check the GPUs.
        yield sse_message(
            f"server with server_id={node_object.metadata.uid} now tracked in database, provisioning verification resources...",
        )

        if server_args.agent_api:
            # If adding a standalone cluster, need to start monitoring
            await start_server_monitoring(server_args.agent_api)

            yield sse_message(
                f"Started monitoring server_id={node_object.metadata.uid}...",
            )

        strategy = await VerificationStrategy.create(node, server_args, server)

        task = asyncio.create_task(strategy.run())

        async for msg in strategy.stream_messages():
            yield msg

        await task

    except Exception as exc:
        error_message = (
            f"unhandled exception bootstrapping new node: {exc}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        yield sse_message(error_message)
        if strategy:
            await strategy.cleanup(delete_node=True)
        await _cleanup(delete_node=True)
        raise
    finally:
        if strategy:
            await strategy.cleanup(delete_node=False)
        await _cleanup(delete_node=False)

    # Astonishing, everything worked.
    async with get_session() as session:
        await session.execute(
            update(GPU).where(GPU.server_id == node_object.metadata.uid).values({"verified": True})
        )
        await session.commit()
    yield sse_message(f"completed server bootstrapping in {time.time() - started_at} seconds!")
