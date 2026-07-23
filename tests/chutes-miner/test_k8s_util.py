import pytest
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


def _make_inputs(version: str, tee: bool = False):
    chute = SimpleNamespace(
        chute_id="chute-123",
        version=version,
        chutes_version=version,
        ref_str="gh://chutes/test",
        filename="main.py",
        image="parachutes/test:latest",
        gpu_count=1,
        tee=tee,
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


def _build_job(version: str, vm_version: str = None):
    chute, server, service = _make_inputs(version)
    return build_chute_job(
        deployment_id="deploy-1",
        chute=chute,
        server=server,
        service=service,
        gpu_uuids=["UUID-1"],
        probe_port=8000,
        vm_version=vm_version,
    )


def _image(job):
    return job.spec.template.spec.containers[0].image


def test_build_chute_job_uses_proxy_when_vm_version_unknown():
    # No VM version reported → fall back to the legacy in-VM proxy hostname.
    image = _image(_build_job("0.6.0"))
    assert image.startswith("validator.localregistry.chutes.ai:")
    assert image.endswith("/parachutes/test:latest")


@pytest.mark.parametrize("vm_version", ["1.3.9", "1.3.0.rc5", "0.9.0"])
def test_build_chute_job_uses_proxy_for_pre_mtls_vm(vm_version):
    # VMs below MTLS_REGISTRY_MIN_VERSION (default 1.4.0) keep using the proxy.
    image = _image(_build_job("0.6.0", vm_version=vm_version))
    assert image.startswith("validator.localregistry.chutes.ai:")
    assert image.endswith("/parachutes/test:latest")


# semcomp compares only the X.Y.Z prefix, so an rc build of 1.4.0 gates on 1.4.0 (direct).
@pytest.mark.parametrize("vm_version", ["1.4.0", "1.4.0.rc1", "1.5.2"])
def test_build_chute_job_uses_direct_registry_for_mtls_vm(vm_version):
    # VMs at or above MTLS_REGISTRY_MIN_VERSION pull direct from registry.chutes.ai over mTLS.
    image = _image(_build_job("0.6.0", vm_version=vm_version))
    assert image == "registry.chutes.ai/parachutes/test:latest"


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

@pytest.mark.parametrize("version", ["0.4.0.rc2", "0.4.0.rc16", "0.4.49.rc100"])
def test_build_chute_job_skips_code_volume_for_newer_rc_version(version):
    job = _build_job(version)
    volumes = job.spec.template.spec.volumes
    mounts = job.spec.template.spec.containers[0].volume_mounts

    assert all(volume.name != "code" for volume in volumes)
    assert all(mount.name != "code" for mount in mounts)


def test_build_chute_job_skips_code_volume_for_tee_chute():
    chute, server, service = _make_inputs("0.3.0", tee=True)
    job = build_chute_job(
        deployment_id="deploy-1",
        chute=chute,
        server=server,
        service=service,
        gpu_uuids=["UUID-1"],
        probe_port=8000,
    )
    volumes = job.spec.template.spec.volumes
    mounts = job.spec.template.spec.containers[0].volume_mounts

    assert all(volume.name != "code" for volume in volumes)
    assert all(mount.name != "code" for mount in mounts)


def test_build_chute_job_attaches_code_volume_for_legacy_non_tee():
    chute, server, service = _make_inputs("0.3.0", tee=False)
    job = build_chute_job(
        deployment_id="deploy-1",
        chute=chute,
        server=server,
        service=service,
        gpu_uuids=["UUID-1"],
        probe_port=8000,
    )
    volumes = job.spec.template.spec.volumes
    mounts = job.spec.template.spec.containers[0].volume_mounts

    assert any(volume.name == "code" for volume in volumes)
    assert any(mount.name == "code" for mount in mounts)
