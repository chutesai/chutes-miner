from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import ssl
import time

import aiohttp
import backoff
from chutes_common.settings import Validator
from chutes_miner.api.config import settings
import chutes_common.constants as cst
from chutes_common.auth import sign_request
from chutes_common.k8s import WatchEventType
from chutes_common.schemas.gpu import GPU
from chutes_common.schemas.server import Server, ServerArgs
from chutes_miner.api.config import validator_by_hotkey
from chutes_miner.api.database import get_session
from chutes_miner.api.exceptions import (
    GPUlessServer,
    BootstrapFailure,
    GraValBootstrapFailure,
    NonEmptyServer,
    TEEBootstrapFailure,
)
from chutes_miner.api.k8s.constants import GRAVAL_JOB_PREFIX, GRAVAL_SVC_PREFIX
from chutes_miner.api.k8s.operator import K8sOperator
from chutes_miner.api.util import sse_message
from loguru import logger
from sqlalchemy import select, update
from kubernetes.client import (
    V1Node,
    V1Job,
    V1JobSpec,
    V1Service,
    V1ObjectMeta,
    V1PodTemplateSpec,
    V1PodSpec,
    V1Container,
    V1ResourceRequirements,
    V1ServiceSpec,
    V1ServicePort,
    V1Probe,
    V1ExecAction,
    V1EnvVar,
)


class VerificationStrategy(ABC):
    def __init__(self, node: V1Node, server_args: ServerArgs, server: Server):
        self.node = node
        self.node_ip = self.node.metadata.labels.get("chutes/external-ip")
        self.server_args = server_args
        self.server = server
        self.validator = validator_by_hotkey(server_args.validator)
        self.queue = asyncio.Queue()
        self._finished = False

    @classmethod
    async def create(cls, node: V1Node, server_args: ServerArgs, server: Server):
        """Async factory method."""
        is_tee = node.metadata.labels.get("chutes/tee", "false").lower() == "true"

        if is_tee:
            return TEEVerificationStrategy(node, server_args, server)
        else:
            return GravalVerificationStrategy(node, server_args, server)

    async def emit_message(self, message: str):
        await self.queue.put(sse_message(message))

    async def run(self):
        try:
            await self.prepare_verification_environment()
            await self.gather_gpu_info()
            await self.verify_with_validator()
        finally:
            self._finished = True

    async def stream_messages(self):
        """Yield messages from the queue until run has finished executing."""
        while not self._finished:
            try:
                # Wait for a message with a timeout to avoid hanging indefinitely
                message = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.queue.task_done()
                yield message
            except asyncio.TimeoutError:
                await asyncio.sleep(1)

        # Drain any remaining messages in the queue after finishing
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                self.queue.task_done()
                yield message
            except asyncio.QueueEmpty:
                break

    @abstractmethod
    async def prepare_verification_environment(self):
        """
        Prepare the environment for verification.
        - Graval: Deploy graval job/service
        - TEE: Verify TEE service availability

        Returns environment details needed for subsequent steps.
        """
        pass

    async def gather_gpu_info(self):
        """
        Gather GPU information using the appropriate method.
        - Graval: Query graval service
        - TEE: Use TDX quotes and nv trust evidence
        """
        raw_device_info = await self._gather_gpu_info()

        self.gpus = await self._track_gpus(raw_device_info)

        model_name = self.gpus[0].device_info["name"]
        await self.emit_message(
            f"discovered {len(self.gpus)} GPUs [{model_name=}] on node, advertising node to {len(settings.validators)} validator(s)...",
        )

    async def _track_gpus(self, devices) -> list[GPU]:
        # Store inventory.
        server_id = self.server.server_id
        validator = self.validator.hotkey
        gpu_short_ref = self.node.metadata.labels.get("gpu-short-ref")
        gpus = []
        async with get_session() as session:
            for device_id in range(len(devices)):
                device_info = devices[device_id]
                gpu = GPU(
                    server_id=server_id,
                    validator=validator,
                    gpu_id=device_info["uuid"],
                    device_info=device_info,
                    model_short_ref=gpu_short_ref,
                    verified=False,
                )
                session.add(gpu)
                gpus.append(gpu)
            await session.commit()
            for idx in range(len(gpus)):
                await session.refresh(gpus[idx])
        return gpus

    @abstractmethod
    async def _gather_gpu_info(self):
        """
        Gather GPU information using the appropriate method.
        - Graval: Query graval service
        - TEE: Use TDX quotes and nv trust evidence
        """
        pass

    @abstractmethod
    async def verify_with_validator(self):
        """
        Advertise nodes to validator using appropriate endpoint.
        - Graval: POST to /nodes (or current endpoint)
        - TEE: POST to /nodes/tee (or similar TEE-specific endpoint)

        Returns (task_id, validator_nodes)
        """
        pass

    @abstractmethod
    async def cleanup(self, delete_node: bool = True) -> None:
        """
        Clean up verification-specific resources.
        - Graval: Remove graval job/service
        - TEE: Cleanup TEE-specific resources if needed
        """
        pass


class GravalVerificationStrategy(VerificationStrategy):
    async def prepare_verification_environment(self):
        """
        Prepare the environment for verification.
        - Graval: Deploy graval job/service
        - TEE: Verify TEE service availability

        Returns environment details needed for subsequent steps.
        """
        graval_job, graval_svc = await self._deploy_graval(
            self.node,
            self.server_args.validator,
            self.server.cpu_per_gpu,
            self.server.memory_per_gpu,
        )

        self.graval_job = graval_job
        self.graval_svc = graval_svc

        await self.emit_message("graval bootstrap job/service created, gathering device info...")

    async def _deploy_graval(
        self, node_object: V1Node, validator_hotkey: str, cpu_per_gpu: int, memory_per_gpu: int
    ):
        """
        Create a job of the GraVal base validation service on a node.
        """
        node_name = node_object.metadata.name
        node_labels = node_object.metadata.labels or {}

        # Double check that we don't already have chute deployments.
        existing_jobs = K8sOperator().get_jobs(label_selector="chute/chute=true,app=graval")
        if any(
            [job for job in existing_jobs.items if job.spec.template.spec.node_name == node_name]
        ):
            raise NonEmptyServer(
                f"Kubernetes node {node_name} already has one or more chute and/or graval jobs."
            )

        # Make sure the GPU labels are set.
        gpu_count = node_labels.get("nvidia.com/gpu.count", "0")
        if not gpu_count or not gpu_count.isdigit() or not 0 < (gpu_count := int(gpu_count)) <= 10:
            raise GPUlessServer(
                f"Kubernetes node {node_name} nvidia.com/gpu.count label missing or invalid: {node_labels.get('nvidia.com/gpu.count')}"
            )

        # Create the job.
        nice_name = node_name.replace(".", "-")
        job = V1Job(
            metadata=V1ObjectMeta(
                name=f"{GRAVAL_JOB_PREFIX}-{nice_name}",
                labels={
                    "app": "graval",
                    "chute/chute": "false",
                    "graval-node": node_name,
                },
            ),
            spec=V1JobSpec(
                parallelism=1,
                completions=1,
                backoff_limit=3,
                ttl_seconds_after_finished=300,
                template=V1PodTemplateSpec(
                    metadata=V1ObjectMeta(labels={"app": "graval", "graval-node": node_name}),
                    spec=V1PodSpec(
                        restart_policy="OnFailure",
                        node_name=node_name,
                        runtime_class_name=settings.nvidia_runtime,
                        containers=[
                            V1Container(
                                name="graval",
                                image=settings.graval_bootstrap_image,
                                image_pull_policy="Always",
                                env=[
                                    V1EnvVar(
                                        name="VALIDATOR_WHITELIST",
                                        value=validator_hotkey,
                                    ),
                                    V1EnvVar(
                                        name="MINER_HOTKEY_SS58",
                                        value=settings.miner_ss58,
                                    ),
                                ],
                                resources=V1ResourceRequirements(
                                    requests={
                                        "cpu": str(gpu_count * cpu_per_gpu),
                                        "memory": f"{int(gpu_count * memory_per_gpu)}Gi",
                                        "nvidia.com/gpu": str(gpu_count),
                                    },
                                    limits={
                                        "cpu": str(gpu_count * cpu_per_gpu),
                                        "memory": f"{int(gpu_count * memory_per_gpu)}Gi",
                                        "nvidia.com/gpu": str(gpu_count),
                                    },
                                ),
                                ports=[{"containerPort": 8000}],
                                readiness_probe=V1Probe(
                                    _exec=V1ExecAction(
                                        command=[
                                            "/bin/sh",
                                            "-c",
                                            "curl -f http://127.0.0.1:8000/ping || exit 1",
                                        ]
                                    ),
                                    initial_delay_seconds=15,
                                    period_seconds=10,
                                    timeout_seconds=1,
                                    success_threshold=1,
                                    failure_threshold=3,
                                ),
                            )
                        ],
                    ),
                ),
            ),
        )

        # And the service that exposes it.
        service = V1Service(
            metadata=V1ObjectMeta(
                name=f"{GRAVAL_SVC_PREFIX}-{nice_name}",
                labels={"app": "graval", "graval-node": node_name},
            ),
            spec=V1ServiceSpec(
                type="NodePort",
                selector={"app": "graval", "graval-node": node_name},
                ports=[V1ServicePort(port=8000, target_port=8000, protocol="TCP")],
            ),
        )

        # Deploy!
        return await K8sOperator().deploy_graval(node_object, job, service)

    async def _gather_gpu_info(
        self,
    ) -> list[GPU]:
        """
        Wait for the graval bootstrap job to be ready, then gather the device info.
        """
        node_object = self.node
        job_name = self.graval_job.metadata.name
        namespace = self.graval_job.metadata.namespace or "chutes"
        expected_gpu_count = int(node_object.metadata.labels.get("nvidia.com/gpu.count", "0"))
        gpu_short_ref = node_object.metadata.labels.get("gpu-short-ref")

        if not gpu_short_ref:
            raise GraValBootstrapFailure("Node does not have required gpu-short-ref label!")

        # Wait for the bootstrap job's pod to be ready.
        start_time = time.time()
        pod_ready = False
        try:
            for event in K8sOperator().watch_pods(
                namespace=namespace,
                label_selector=f"job-name={job_name}",
                timeout=settings.graval_bootstrap_timeout,
            ):
                pod = event.object
                if event.type == WatchEventType.DELETED:
                    continue
                if pod.status.phase == "Failed":
                    raise GraValBootstrapFailure(f"Bootstrap pod failed: {pod.status.message}")
                if pod.status.phase == "Running":
                    if pod.status.container_statuses:
                        all_ready = all(cs.ready for cs in pod.status.container_statuses)
                        if all_ready:
                            pod_ready = True
                            break
                if (delta := time.time() - start_time) >= settings.graval_bootstrap_timeout:
                    raise TimeoutError(f"GraVal bootstrap job not ready after {delta} seconds!")
                await asyncio.sleep(1)
        except Exception as exc:
            raise GraValBootstrapFailure(f"Error waiting for graval bootstrap job: {exc}")
        if not pod_ready:
            raise GraValBootstrapFailure("GraVal bootstrap job never reached ready state.")

        # Configure our validation host/port.
        node_port = None
        node_ip = node_object.metadata.labels.get("chutes/external-ip")
        for port in self.graval_svc.spec.ports:
            if port.node_port:
                node_port = port.node_port
                break

        # Query the GPU information.
        devices = None
        try:
            devices = await self._fetch_devices(f"http://{node_ip}:{node_port}/devices")
            assert devices
            assert len(devices) == expected_gpu_count
        except Exception as exc:
            raise GraValBootstrapFailure(
                f"Failed to fetch devices from GraVal bootstrap: {node_ip}:{node_port}/devices: {exc}"
            )

        return devices

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=5,
    )
    async def _fetch_devices(self, url):
        """
        Query the GraVal bootstrap API for device info.
        """
        nonce = str(int(time.time()))
        headers = {
            cst.MINER_HEADER: settings.miner_ss58,
            cst.VALIDATOR_HEADER: settings.miner_ss58,
            cst.NONCE_HEADER: nonce,
        }
        headers[cst.SIGNATURE_HEADER] = settings.miner_keypair.sign(
            ":".join([settings.miner_ss58, settings.miner_ss58, nonce, "graval"])
        ).hex()
        logger.debug(f"Authenticating: {headers}")
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(url, headers=headers, timeout=5) as response:
                return (await response.json())["devices"]

    async def verify_with_validator(self):
        """
        Advertise nodes to validator using appropriate endpoint.
        - Graval: POST to /nodes (or current endpoint)
        - TEE: POST to /nodes/tee (or similar TEE-specific endpoint)

        Returns (task_id, validator_nodes)
        """
        await self._advertise_to_validator()

        # Wait for verification from this validator.
        status = await self._wait_for_verification_task()
        if status:
            await self.emit_message(
                f"validator {self.validator.hotkey} has successfully performed verification"
            )
        else:
            error_message = f"Verification failed for {self.validator.hotkey}, aborting!"
            await self.emit_message(error_message)
            raise BootstrapFailure(error_message)

    async def _advertise_to_validator(self):
        """
        Advertise nodes to validator using appropriate endpoint.
        - Graval: POST to /nodes (or current endpoint)
        - TEE: POST to /nodes/tee (or similar TEE-specific endpoint)

        Returns (task_id, validator_nodes)
        """
        seed = None
        validator = self.validator
        await self.emit_message(
            f"advertising node to {validator.hotkey} via {validator.api}...",
        )
        validator_nodes = None
        task_id = None
        try:
            task_id, validator_nodes = await self._advertise_nodes(validator, self.gpus)
        except Exception as exc:
            await self.emit_message(
                f"failed to advertising node to {validator.hotkey} via {validator.api}: {exc}",
            )
            raise
        assert len(set(node["seed"] for node in validator_nodes)) == 1, (
            f"more than one seed produced from {validator.hotkey}!"
        )
        if not seed:
            seed = validator_nodes[0]["seed"]
        else:
            assert seed == validator_nodes[0]["seed"], (
                f"validators produced differing seeds {seed} vs {validator_nodes[0]['seed']}"
            )
        await self.emit_message(
            f"successfully advertised node {self.node.metadata.uid} to validator {validator.hotkey}, received seed: {seed}"
        )

        async with get_session() as session:
            await session.execute(
                update(Server)
                .where(Server.server_id == self.node.metadata.uid)
                .values({"seed": seed})
            )
            await session.commit()

        self.task_id = task_id
        self.validator_nodes = validator_nodes

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=5,
    )
    async def _advertise_nodes(self, validator: Validator, gpus: list[GPU]):
        """
        Post GPU information to one validator, with retries.
        """
        async with aiohttp.ClientSession() as session:
            device_infos = [
                {
                    **gpus[idx].device_info,
                    **dict(
                        device_index=idx,
                        gpu_identifier=gpus[idx].model_short_ref,
                        verification_host=gpus[idx].server.ip_address,
                        verification_port=gpus[idx].server.verification_port,
                    ),
                }
                for idx in range(len(gpus))
            ]
            headers, payload_string = sign_request(
                payload={"nodes": device_infos, "server_id": gpus[0].server_id}
            )
            async with session.post(
                f"{validator.api}/nodes/", data=payload_string, headers=headers
            ) as response:
                response_text = await response.text()
                assert response.status == 202, response_text
                data = await response.json()
                nodes = data.get("nodes")
                task_id = data.get("task_id")
                assert len(nodes) == len(gpus)
                assert task_id
                logger.success(
                    f"Successfully advertised {len(gpus)} to {validator.hotkey} via {validator.api}"
                )
                return task_id, nodes

    async def _wait_for_verification_task(self):
        """
        Wait for the verification task on the validator to complete
        """
        while (status := await self._check_verification_task_status()) is None:
            await self.emit_message(
                f"waiting for validator {self.validator.hotkey} to finish GPU verification..."
            )
            await asyncio.sleep(1)
        return status

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=5,
    )
    async def _check_verification_task_status(self) -> bool:
        """
        Check the GPU verification task status.
        """
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            headers, _ = sign_request(purpose="graval")
            async with session.get(
                f"{self.validator.api}/nodes/verification_status",
                params={"task_id": self.task_id},
                headers=headers,
            ) as response:
                data = await response.json()
                if (status := data.get("status")) == "pending":
                    return None
                if status in ["error", "failed"]:
                    return False
                return True

    async def cleanup(self, delete_node: bool = True) -> None:
        """
        Clean up verification-specific resources.
        - Graval: Remove graval job/service
        - TEE: Cleanup TEE-specific resources if needed
        """
        node_object = self.node

        await K8sOperator().cleanup_graval(node_object)

        if delete_node:
            node_uid = node_object.metadata.uid
            node_name = node_object.metadata.name
            logger.info(f"Purging failed server: {node_name=} {node_uid=}")
            validator = self.validator
            server_id = None

            async with get_session() as session:
                server = (
                    (await session.execute(select(Server).where(Server.server_id == node_uid)))
                    .unique()
                    .scalar_one_or_none()
                )
                if server:
                    server_id = server.server_id
                    await session.delete(server)
                await session.commit()

            if server_id:
                try:
                    async with aiohttp.ClientSession(raise_for_status=True) as http_session:
                        headers, _ = sign_request(purpose="tee")
                        async with http_session.delete(
                            f"{validator.api}/servers/{server_id}", headers=headers
                        ) as resp:
                            logger.success(
                                f"Successfully purged {server_id=} from validator={validator.hotkey}: {await resp.json()}"
                            )
                except Exception as exc:
                    logger.warning(
                        f"Error purging {server_id=} from validator={validator.hotkey}: {exc}"
                    )
            else:
                logger.warning(
                    "Unable to purge validator server entry because server record was not found",
                )


class TEEVerificationStrategy(VerificationStrategy):
    @asynccontextmanager
    async def _attestation_session(self):
        """
        Creates an aiohttp session configured for the attestation service.

        SSL verification is disabled because certificate authenticity is verified
        through TDX quotes, which include a hash of the service's public key.
        """
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector, raise_for_status=True) as session:
            yield session

    async def prepare_verification_environment(self):
        """
        Prepare the environment for verification.
        - Graval: Deploy graval job/service
        - TEE: Verify TEE service availability

        Returns environment details needed for subsequent steps.
        """
        await self._verify_attestation_service()

    async def _verify_attestation_service(self):
        try:
            async with self._attestation_session() as http_session:
                async with http_session.get(f"https://{self.node_ip}:30443/server/health"):
                    logger.success(f"Verified attestation service for {self.server.name}")
                    await self.emit_message(
                        f"Verified attestation service is available for {self.server.name}[{self.node_ip}]"
                    )
        except Exception as exc:
            logger.warning(f"Error verifying attestation services for {self.server.name}: {exc}")
            raise TEEBootstrapFailure(f"Failed to verify attestion service for {self.server.name}")

    async def _gather_gpu_info(self):
        """
        Wait for the graval bootstrap job to be ready, then gather the device info.
        """
        # Query the GPU information.
        devices = None
        expected_gpu_count = int(self.node.metadata.labels.get("nvidia.com/gpu.count", "0"))
        try:
            raw_devices = await self._fetch_devices()
            assert raw_devices
            assert len(raw_devices) == expected_gpu_count

            devices = raw_devices

        except Exception as exc:
            raise TEEBootstrapFailure(
                f"Failed to fetch devices from attestation service: {self.node_ip}:30443/server/devices: {exc}"
            )

        return devices

    @backoff.on_exception(
        backoff.constant,
        Exception,
        jitter=None,
        interval=3,
        max_tries=5,
    )
    async def _fetch_devices(self):
        """
        Query the GraVal bootstrap API for device info.
        """
        devices = []
        async with self._attestation_session() as http_session:
            headers, _ = sign_request(purpose="attest")
            async with http_session.get(
                f"https://{self.node_ip}:30443/server/devices", headers=headers
            ) as resp:
                devices = await resp.json()
                logger.success(f"Retrieved {len(devices)} GPUs from {self.server.name}.")

        return devices

    async def verify_with_validator(self):
        """
        Advertise nodes to validator using appropriate endpoint.
        - Graval: POST to /nodes (or current endpoint)
        - TEE: POST to /nodes/tee (or similar TEE-specific endpoint)

        Returns (task_id, validator_nodes)
        """
        validator = self.validator
        await self.emit_message(
            f"Verifying server with {validator.hotkey} via {validator.api}...",
        )

        try:
            await self._advertise_server()
        except Exception as exc:
            await self.emit_message(
                f"failed to verify server with {validator.hotkey} via {validator.api}: {exc}",
            )
            raise

        await self.emit_message(
            f"successfully verifeid server {self.server.name} to validator {validator.hotkey}"
        )

    @backoff.on_exception(
        backoff.constant,
        (aiohttp.ClientError, asyncio.TimeoutError),
        giveup=lambda e: isinstance(e, aiohttp.ClientResponseError),
        jitter=None,
        interval=3,
        max_tries=5,
    )
    async def _advertise_server(self):
        """
        Post Server information to one validator, with retries.
        """
        gpus = self.gpus
        async with aiohttp.ClientSession() as session:
            device_infos = [
                {
                    **gpus[idx].device_info,
                    **dict(
                        device_index=idx,
                        gpu_identifier=gpus[idx].model_short_ref,
                        verification_host=self.node_ip,
                        verification_port="30443",
                    ),
                }
                for idx in range(len(gpus))
            ]
            headers, payload_string = sign_request(
                payload={
                    "id": self.server.server_id,
                    "host": self.node_ip,
                    "gpus": device_infos,
                }
            )
            async with session.post(
                f"{self.validator.api}/servers/", data=payload_string, headers=headers
            ) as response:
                response_text = await response.text()
                if response.status != 201:
                    # Raise ClientResponseError for bad status codes - won't be retried
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=response_text,
                        headers=response.headers,
                    )
                logger.success(
                    f"Successfully advertised {self.server.name} with {len(gpus)} GPUs to {self.validator.hotkey} via {self.validator.api}"
                )

    async def cleanup(self, delete_node: bool = True) -> None:
        """
        Clean up verification-specific resources.
        - Graval: Remove graval job/service
        - TEE: Cleanup TEE-specific resources if needed
        """
        node_object = self.node

        if delete_node:
            node_uid = node_object.metadata.uid
            logger.info(f"Purging failed server: {self.server.name=} {node_uid=}")
            validator = self.validator
            async with get_session() as session:
                server_id = None
                server = (
                    (await session.execute(select(Server).where(Server.server_id == node_uid)))
                    .unique()
                    .scalar_one_or_none()
                )
                if server:
                    server_id = server.server_id
                    await session.delete(server)
                await session.commit()

                try:
                    async with aiohttp.ClientSession(raise_for_status=True) as http_session:
                        headers, _ = sign_request(purpose="tee")
                        async with http_session.delete(
                            f"{validator.api}/servers/{server_id}", headers=headers
                        ) as resp:
                            logger.success(
                                f"Successfully purged {server_id=} from validator={validator.hotkey}: {await resp.json()}"
                            )
                except Exception as exc:
                    logger.warning(
                        f"Error purging {server_id=} from validator={validator.hotkey}: {exc}"
                    )
