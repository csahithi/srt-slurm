# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for endpoint allocation logic."""

import pytest

from srtctl.core.topology import allocate_endpoints, endpoints_to_processes


class TestAllocateEndpoints:
    """Tests for allocate_endpoints function."""

    def test_multiple_endpoints_per_node(self):
        """Test multiple endpoints sharing a single node."""
        # 2 prefill endpoints, 2 GPUs each, 4 GPUs per node -> both on node0
        # 2 decode endpoints, 2 GPUs each, 4 GPUs per node -> both on node1
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        assert len(endpoints) == 4

        # Check prefill endpoints - both should be on the SAME node
        prefill_eps = [e for e in endpoints if e.mode == "prefill"]
        assert len(prefill_eps) == 2
        assert prefill_eps[0].nodes[0] == prefill_eps[1].nodes[0] == "node0"
        assert prefill_eps[0].total_gpus == 2
        assert prefill_eps[1].total_gpus == 2
        # They should have different GPU indices
        assert prefill_eps[0].gpu_indices != prefill_eps[1].gpu_indices
        assert prefill_eps[0].gpu_indices == frozenset({0, 1})
        assert prefill_eps[1].gpu_indices == frozenset({2, 3})

        # Check decode endpoints - both should be on the SAME node (node1)
        decode_eps = [e for e in endpoints if e.mode == "decode"]
        assert len(decode_eps) == 2
        assert decode_eps[0].nodes[0] == decode_eps[1].nodes[0] == "node1"
        assert decode_eps[0].gpu_indices == frozenset({0, 1})
        assert decode_eps[1].gpu_indices == frozenset({2, 3})

    def test_full_node_endpoints(self):
        """Test endpoints that use full nodes."""
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=4,
            gpus_per_decode=4,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1", "node2", "node3"),
        )

        assert len(endpoints) == 4

        # Each endpoint should use a full node
        for ep in endpoints:
            assert ep.total_gpus == 4
            assert len(ep.nodes) == 1

    def test_multi_node_endpoints(self):
        """Test endpoints that span multiple nodes."""
        # 1 prefill worker, 8 GPUs, 4 GPUs per node -> spans 2 nodes
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=8,
            gpus_per_decode=8,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1", "node2", "node3"),
        )

        assert len(endpoints) == 2

        # Each endpoint should span 2 nodes
        for ep in endpoints:
            assert len(ep.nodes) == 2
            assert ep.total_gpus == 8

    def test_insufficient_gpus(self):
        """Test that we raise an error when there are insufficient GPUs."""
        # This should raise an IndexError when trying to access nodes that don't exist
        with pytest.raises((ValueError, IndexError)):
            allocate_endpoints(
                num_prefill=2,
                num_decode=2,
                num_agg=0,
                gpus_per_prefill=8,
                gpus_per_decode=8,
                gpus_per_agg=8,
                gpus_per_node=4,
                available_nodes=("node0", "node1"),  # Only 8 GPUs total, need 32
            )

    def test_single_endpoint_single_gpu(self):
        """Test edge case: single endpoint with single GPU."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=1,
            gpus_per_decode=1,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        assert len(endpoints) == 1
        assert endpoints[0].mode == "prefill"
        assert endpoints[0].total_gpus == 1

    def test_aggregated_mode(self):
        """Test aggregated mode (no disaggregation)."""
        endpoints = allocate_endpoints(
            num_prefill=0,
            num_decode=0,
            num_agg=2,
            gpus_per_prefill=4,
            gpus_per_decode=4,
            gpus_per_agg=4,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        assert len(endpoints) == 2
        for ep in endpoints:
            assert ep.mode == "agg"
            assert ep.total_gpus == 4


class TestEndpointsToProcesses:
    """Tests for endpoints_to_processes function."""

    def test_process_construction(self):
        """Test that endpoints_to_processes creates correct process mappings."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        processes = endpoints_to_processes(endpoints, base_port=8081)

        # SGLang creates one process per node
        assert len(processes) == 2

        # Check all sys_ports are unique
        ports = [p.sys_port for p in processes]
        assert len(ports) == len(set(ports)), "All processes should have unique sys_ports"

    def test_multi_node_process_construction(self):
        """Test process construction for multi-node endpoints."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=8,
            gpus_per_decode=4,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        processes = endpoints_to_processes(endpoints, base_port=8081)

        # Multi-node endpoint should create one process per node
        assert len(processes) == 2
        nodes = [p.node for p in processes]
        assert "node0" in nodes
        assert "node1" in nodes

    def test_cuda_visible_devices(self):
        """Test that CUDA_VISIBLE_DEVICES is set correctly for each process."""
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        processes = endpoints_to_processes(endpoints, base_port=8081)

        # Each process should have correct GPU indices
        for p in processes:
            assert len(p.gpu_indices) == 2
            # Check cuda_visible_devices is formatted correctly
            assert "," in p.cuda_visible_devices or p.cuda_visible_devices.isdigit()
