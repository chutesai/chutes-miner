"""
Tests for node_selector normalization, stable hashing,
and gpu_count parameter threading through the deploy pipeline.
"""

from types import SimpleNamespace

from kubernetes.client import V1Service, V1ServiceSpec, V1ServicePort

from chutes_miner.gepetto import (
    normalize_node_selector,
    _stable_selector_hash,
)
from chutes_miner.api.k8s.util import build_chute_job


# ---------------------------------------------------------------------------
# normalize_node_selector
# ---------------------------------------------------------------------------


class TestNormalizeNodeSelector:
    def test_flat_dict_becomes_single_element_list(self):
        flat = {"gpu_count": 2, "supported_gpus": ["a100"]}
        result = normalize_node_selector(flat)
        assert result == [flat]
        assert isinstance(result, list)

    def test_list_stays_as_list(self):
        selectors = [
            {"gpu_count": 1, "supported_gpus": ["h200"]},
            {"gpu_count": 4, "supported_gpus": ["l40s"]},
        ]
        result = normalize_node_selector(selectors)
        assert result == selectors

    def test_single_element_list_unchanged(self):
        selectors = [{"gpu_count": 1, "supported_gpus": ["a100"]}]
        result = normalize_node_selector(selectors)
        assert result == selectors

    def test_returns_new_list_from_tuple(self):
        selectors = ({"gpu_count": 1, "supported_gpus": ["a100"]},)
        result = normalize_node_selector(selectors)
        assert result == [{"gpu_count": 1, "supported_gpus": ["a100"]}]
        assert isinstance(result, list)

    def test_flat_dict_with_include(self):
        flat = {"gpu_count": 1, "include": ["h200"], "supported_gpus": ["h200"]}
        result = normalize_node_selector(flat)
        assert result == [flat]

    def test_empty_list(self):
        result = normalize_node_selector([])
        assert result == []


# ---------------------------------------------------------------------------
# build_chute_job gpu_count handling
# ---------------------------------------------------------------------------


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


def _make_chute(node_selector):
    return SimpleNamespace(
        chute_id="chute-123",
        version="0.4.0",
        chutes_version="0.4.0",
        ref_str="gh://chutes/test",
        filename="main.py",
        image="parachutes/test:latest",
        node_selector=node_selector,
        tee=False,
    )


def _make_server(cpu_per_gpu=4, memory_per_gpu=16):
    return SimpleNamespace(
        cpu_per_gpu=cpu_per_gpu,
        memory_per_gpu=memory_per_gpu,
        seed=42,
        validator="Validator",
        name="node-1",
        ip_address="10.0.0.10",
    )


class TestBuildChuteJobGpuCount:
    def test_explicit_gpu_count_overrides_selector(self):
        """When gpu_count is passed explicitly, it should be used for CPU/RAM."""
        chute = _make_chute([{"gpu_count": 1, "supported_gpus": ["a100"]}])
        server = _make_server(cpu_per_gpu=4, memory_per_gpu=16)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2"],
            probe_port=8000,
            gpu_count=2,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "8"  # 4 * 2
        assert resources.requests["memory"] == "32Gi"  # 16 * 2

    def test_fallback_to_first_selector_gpu_count_list(self):
        """When gpu_count is None and node_selector is a list, fall back to [0]."""
        chute = _make_chute(
            [
                {"gpu_count": 3, "supported_gpus": ["a100"]},
                {"gpu_count": 1, "supported_gpus": ["h100"]},
            ]
        )
        server = _make_server(cpu_per_gpu=2, memory_per_gpu=8)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2", "UUID-3"],
            probe_port=8000,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "6"  # 2 * 3
        assert resources.requests["memory"] == "24Gi"  # 8 * 3

    def test_fallback_to_dict_gpu_count(self):
        """When gpu_count is None and node_selector is a flat dict, normalize and use it."""
        chute = _make_chute({"gpu_count": 2, "supported_gpus": ["a100"]})
        server = _make_server(cpu_per_gpu=4, memory_per_gpu=16)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2"],
            probe_port=8000,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "8"  # 4 * 2
        assert resources.requests["memory"] == "32Gi"  # 16 * 2

    def test_gpu_count_1_resources(self):
        """Single GPU allocation."""
        chute = _make_chute([{"gpu_count": 1, "supported_gpus": ["a100"]}])
        server = _make_server(cpu_per_gpu=4, memory_per_gpu=16)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1"],
            probe_port=8000,
            gpu_count=1,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "4"
        assert resources.requests["memory"] == "16Gi"
        assert resources.limits["cpu"] == "4"
        assert resources.limits["memory"] == "16Gi"

    def test_gpu_count_4_resources(self):
        """Multi-GPU allocation."""
        chute = _make_chute([{"gpu_count": 4, "supported_gpus": ["l40s"]}])
        server = _make_server(cpu_per_gpu=2, memory_per_gpu=8)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2", "UUID-3", "UUID-4"],
            probe_port=8000,
            gpu_count=4,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "8"  # 2 * 4
        assert resources.requests["memory"] == "32Gi"  # 8 * 4


# ---------------------------------------------------------------------------
# End-to-end: normalize -> build
# ---------------------------------------------------------------------------


class TestEndToEndNodeSelector:
    """Verify the full flow from raw validator data to job creation."""

    def test_flat_dict_node_selector_flow(self):
        """Flat dict from validator -> normalize -> build job."""
        raw = {"gpu_count": 2, "supported_gpus": ["a100", "h100"]}
        normalized = normalize_node_selector(raw)
        assert normalized == [raw]

        chute = _make_chute(raw)  # store as-is (dict)
        server = _make_server(cpu_per_gpu=4, memory_per_gpu=16)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2"],
            probe_port=8000,
            gpu_count=2,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "8"  # 4 * 2
        assert resources.requests["memory"] == "32Gi"  # 16 * 2

    def test_multi_selector_node_selector_flow(self):
        """List of selectors from validator -> normalize."""
        raw = [
            {"gpu_count": 1, "include": ["h200"], "supported_gpus": ["h200"]},
            {"gpu_count": 4, "include": ["l40s"], "supported_gpus": ["l40s"]},
        ]
        normalized = normalize_node_selector(raw)
        assert normalized == raw

    def test_selector_with_explicit_gpu_count_for_second_selector(self):
        """When deploying with a matched selector, pass its gpu_count explicitly."""
        selectors = [
            {"gpu_count": 1, "supported_gpus": ["h200"]},
            {"gpu_count": 4, "supported_gpus": ["l40s"]},
        ]
        chute = _make_chute(selectors)
        server = _make_server(cpu_per_gpu=2, memory_per_gpu=8)

        # Simulate matching the second selector (l40s with gpu_count=4)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2", "UUID-3", "UUID-4"],
            probe_port=8000,
            gpu_count=4,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "8"  # 2 * 4
        assert resources.requests["memory"] == "32Gi"  # 8 * 4


# ---------------------------------------------------------------------------
# Varying gpu_count across selectors
# ---------------------------------------------------------------------------


class TestVaryingGpuCountSelectors:
    """
    Test the real-world scenario: a chute has multiple selectors with different
    gpu_counts (e.g. 1xH200 OR 4xL40S). The correct gpu_count must be picked
    based on which selector matched, and threaded through to resource allocation.
    """

    SELECTORS = [
        {"gpu_count": 1, "include": ["h200"], "supported_gpus": ["h200"]},
        {"gpu_count": 2, "include": ["a100"], "supported_gpus": ["a100"]},
        {"gpu_count": 4, "include": ["l40s"], "supported_gpus": ["l40s"]},
    ]

    def test_normalization_preserves_all_selectors(self):
        normalized = normalize_node_selector(self.SELECTORS)
        assert len(normalized) == 3
        assert normalized[0]["gpu_count"] == 1
        assert normalized[1]["gpu_count"] == 2
        assert normalized[2]["gpu_count"] == 4

    def test_first_selector_match_1_gpu(self):
        """Deploy on h200 server -> should use gpu_count=1."""
        chute = _make_chute(self.SELECTORS)
        server = _make_server(cpu_per_gpu=8, memory_per_gpu=32)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1"],
            probe_port=8000,
            gpu_count=1,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "8"  # 8 * 1
        assert resources.requests["memory"] == "32Gi"  # 32 * 1

    def test_second_selector_match_2_gpus(self):
        """Deploy on a100 server -> should use gpu_count=2."""
        chute = _make_chute(self.SELECTORS)
        server = _make_server(cpu_per_gpu=8, memory_per_gpu=32)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2"],
            probe_port=8000,
            gpu_count=2,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "16"  # 8 * 2
        assert resources.requests["memory"] == "64Gi"  # 32 * 2

    def test_third_selector_match_4_gpus(self):
        """Deploy on l40s server -> should use gpu_count=4."""
        chute = _make_chute(self.SELECTORS)
        server = _make_server(cpu_per_gpu=4, memory_per_gpu=16)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2", "UUID-3", "UUID-4"],
            probe_port=8000,
            gpu_count=4,
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "16"  # 4 * 4
        assert resources.requests["memory"] == "64Gi"  # 16 * 4

    def test_no_explicit_gpu_count_falls_back_to_first_selector(self):
        """Without explicit gpu_count, first selector's count is used."""
        chute = _make_chute(self.SELECTORS)
        server = _make_server(cpu_per_gpu=8, memory_per_gpu=32)
        job = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1"],
            probe_port=8000,
            # no gpu_count -> falls back to SELECTORS[0]["gpu_count"] == 1
        )
        resources = job.spec.template.spec.containers[0].resources
        assert resources.requests["cpu"] == "8"  # 8 * 1
        assert resources.requests["memory"] == "32Gi"  # 32 * 1

    def test_wrong_gpu_count_produces_wrong_resources(self):
        """
        Demonstrate that passing the wrong selector's gpu_count would
        produce different resources — validating that the parameter matters.
        """
        chute = _make_chute(self.SELECTORS)
        server = _make_server(cpu_per_gpu=4, memory_per_gpu=16)

        # If we matched l40s (gpu_count=4) but mistakenly passed gpu_count=1
        job_wrong = build_chute_job(
            "deploy-1",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2", "UUID-3", "UUID-4"],
            probe_port=8000,
            gpu_count=1,  # wrong!
        )
        job_correct = build_chute_job(
            "deploy-2",
            chute,
            server,
            _make_service(),
            gpu_uuids=["UUID-1", "UUID-2", "UUID-3", "UUID-4"],
            probe_port=8000,
            gpu_count=4,  # correct
        )
        wrong_cpu = job_wrong.spec.template.spec.containers[0].resources.requests["cpu"]
        correct_cpu = job_correct.spec.template.spec.containers[0].resources.requests["cpu"]
        assert wrong_cpu == "4"  # 4 * 1
        assert correct_cpu == "16"  # 4 * 4
        assert wrong_cpu != correct_cpu

    def test_selector_matching_logic(self):
        """
        Simulate the matching logic used in rolling_update:
        iterate selectors, find one whose supported_gpus contains the server's GPU type,
        and extract its gpu_count.
        """
        normalized = normalize_node_selector(self.SELECTORS)

        server_gpu_type = "a100"
        matched_gpu_count = None
        for selector in normalized:
            if server_gpu_type in selector.get("supported_gpus", []):
                matched_gpu_count = selector["gpu_count"]
                break
        assert matched_gpu_count == 2

        server_gpu_type = "l40s"
        matched_gpu_count = None
        for selector in normalized:
            if server_gpu_type in selector.get("supported_gpus", []):
                matched_gpu_count = selector["gpu_count"]
                break
        assert matched_gpu_count == 4

        server_gpu_type = "h200"
        matched_gpu_count = None
        for selector in normalized:
            if server_gpu_type in selector.get("supported_gpus", []):
                matched_gpu_count = selector["gpu_count"]
                break
        assert matched_gpu_count == 1

    def test_selector_matching_unknown_gpu_type_returns_none(self):
        """If the server GPU type isn't in any selector, no match is found."""
        normalized = normalize_node_selector(self.SELECTORS)
        server_gpu_type = "t4"
        matched_gpu_count = None
        for selector in normalized:
            if server_gpu_type in selector.get("supported_gpus", []):
                matched_gpu_count = selector["gpu_count"]
                break
        assert matched_gpu_count is None

    def test_flat_dict_selector_matching(self):
        """When node_selector is a flat dict, normalize first then match."""
        flat = {"gpu_count": 2, "supported_gpus": ["a100"]}
        normalized = normalize_node_selector(flat)
        assert len(normalized) == 1
        matched = None
        for selector in normalized:
            if "a100" in selector.get("supported_gpus", []):
                matched = selector["gpu_count"]
                break
        assert matched == 2


# ---------------------------------------------------------------------------
# _stable_selector_hash: deterministic hashing
# ---------------------------------------------------------------------------


class TestStableSelectorHash:
    """Ensure hash is deterministic regardless of key insertion order."""

    def test_same_data_same_hash(self):
        a = [{"gpu_count": 1, "supported_gpus": ["h200"]}]
        b = [{"gpu_count": 1, "supported_gpus": ["h200"]}]
        assert _stable_selector_hash(a) == _stable_selector_hash(b)

    def test_different_key_order_same_hash(self):
        """JSONB may reorder keys. sorted JSON normalizes this."""
        from collections import OrderedDict

        a = [OrderedDict([("gpu_count", 1), ("supported_gpus", ["h200"])])]
        b = [OrderedDict([("supported_gpus", ["h200"]), ("gpu_count", 1)])]
        assert _stable_selector_hash(a) == _stable_selector_hash(b)

    def test_different_data_different_hash(self):
        a = [{"gpu_count": 1, "supported_gpus": ["h200"]}]
        b = [{"gpu_count": 4, "supported_gpus": ["l40s"]}]
        assert _stable_selector_hash(a) != _stable_selector_hash(b)

    def test_multi_selector_order_matters(self):
        """Different selector ordering = different hash (order is semantically meaningful)."""
        a = [
            {"gpu_count": 1, "supported_gpus": ["h200"]},
            {"gpu_count": 4, "supported_gpus": ["l40s"]},
        ]
        b = [
            {"gpu_count": 4, "supported_gpus": ["l40s"]},
            {"gpu_count": 1, "supported_gpus": ["h200"]},
        ]
        assert _stable_selector_hash(a) != _stable_selector_hash(b)

    def test_extra_keys_included(self):
        """Extra keys like 'include' affect the hash."""
        a = [{"gpu_count": 1, "supported_gpus": ["h200"]}]
        b = [{"gpu_count": 1, "include": ["h200"], "supported_gpus": ["h200"]}]
        assert _stable_selector_hash(a) != _stable_selector_hash(b)

    def test_dict_and_list_produce_same_hash(self):
        """A flat dict and a list wrapping it should hash the same (both normalize to list)."""
        flat = {"gpu_count": 1, "supported_gpus": ["h200"]}
        as_list = [{"gpu_count": 1, "supported_gpus": ["h200"]}]
        assert _stable_selector_hash(flat) == _stable_selector_hash(as_list)
