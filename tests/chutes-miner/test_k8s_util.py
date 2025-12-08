from types import SimpleNamespace

from kubernetes.client import V1Service, V1ServiceSpec, V1ServicePort

from chutes_miner.api.k8s.util import build_chute_job


def _make_service() -> V1Service:
    return V1Service(
        spec=V1ServiceSpec(
            type="NodePort",
            selector={"app": "chute"},
            external_traffic_policy="Local",
            ports=[
                V1ServicePort(port=8000, target_port=8000, node_port=30080, protocol="TCP"),
                V1ServicePort(port=8001, target_port=8001, node_port=30081, protocol="TCP"),
            ],
        )
    )


def _make_inputs(version: str):
    chute = SimpleNamespace(
        chute_id="chute-123",
        version=version,
        chutes_version=version,
        ref_str="gh://chutes/test",
        filename="main.py",
        image="parachutes/test:latest",
        gpu_count=1,
    )
    server = SimpleNamespace(
        cpu_per_gpu=1,
        memory_per_gpu=2,
        seed=42,
        validator="Validator",
        name="node-1",
        ip_address="10.0.0.10",
    )
    service = _make_service()
    return chute, server, service


def _build_job(version: str):
    chute, server, service = _make_inputs(version)
    return build_chute_job(
        deployment_id="deploy-1",
        chute=chute,
        server=server,
        service=service,
        gpu_uuids=["UUID-1"],
        probe_port=8000,
    )


def test_build_chute_job_attaches_code_volume_for_legacy_version():
    job = _build_job("0.3.59")
    volumes = job.spec.template.spec.volumes
    mounts = job.spec.template.spec.containers[0].volume_mounts

    assert any(volume.name == "code" for volume in volumes)
    assert any(mount.name == "code" for mount in mounts)


def test_build_chute_job_skips_code_volume_for_min_version():
    job = _build_job("0.3.61")
    volumes = job.spec.template.spec.volumes
    mounts = job.spec.template.spec.containers[0].volume_mounts

    assert all(volume.name != "code" for volume in volumes)
    assert all(mount.name != "code" for mount in mounts)

def test_build_chute_job_skips_code_volume_for_newer_version():
    job = _build_job("0.3.65")
    volumes = job.spec.template.spec.volumes
    mounts = job.spec.template.spec.containers[0].volume_mounts

    assert all(volume.name != "code" for volume in volumes)
    assert all(mount.name != "code" for mount in mounts)
