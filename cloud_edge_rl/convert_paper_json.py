#!/usr/bin/env python3
"""Convert paper JSON graphs into Cloud–Edge–Device operator graphs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DeviceSpec:
    device_id: int
    name: str
    layer: str
    compute_capacity: float
    energy_budget: float
    trust_level: float
    parent_edge: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Paper-format JSON graph")
    parser.add_argument("--output", required=True, type=Path, help="Output Cloud–Edge–Device JSON")
    parser.add_argument("--edges", type=int, default=1, help="Number of edge servers")
    parser.add_argument("--devices-per-edge", type=int, default=1, help="Devices per edge server")
    parser.add_argument("--base-latency", choices=["cpuLatency", "fpgaLatency"], default="cpuLatency")
    parser.add_argument("--cloud-scale", type=float, default=0.5)
    parser.add_argument("--edge-scale", type=float, default=1.0)
    parser.add_argument("--device-scale", type=float, default=2.0)
    parser.add_argument("--cloud-energy", type=float, default=0.08, help="Energy cost per ms")
    parser.add_argument("--edge-energy", type=float, default=0.15, help="Energy cost per ms")
    parser.add_argument("--device-energy", type=float, default=0.3, help="Energy cost per ms")
    parser.add_argument("--trust-default", type=float, default=0.6)
    parser.add_argument("--cloud-trust", type=float, default=0.95)
    parser.add_argument("--edge-trust", type=float, default=0.85)
    parser.add_argument("--device-trust", type=float, default=0.7)
    parser.add_argument("--cloud-capacity", type=float, default=1000)
    parser.add_argument("--edge-capacity", type=float, default=400)
    parser.add_argument("--device-capacity", type=float, default=100)
    parser.add_argument("--cloud-energy-budget", type=float, default=100000)
    parser.add_argument("--edge-energy-budget", type=float, default=30000)
    parser.add_argument("--device-energy-budget", type=float, default=5000)
    parser.add_argument("--cloud-edge-bandwidth", type=float, default=1e9)
    parser.add_argument("--cloud-edge-latency", type=float, default=8)
    parser.add_argument("--edge-device-bandwidth", type=float, default=5e8)
    parser.add_argument("--edge-device-latency", type=float, default=1)
    parser.add_argument("--device-device-bandwidth", type=float, default=8e8)
    parser.add_argument("--device-device-latency", type=float, default=0.5)
    parser.add_argument("--cloud-multiplier", type=float, default=2.0, help="Multiply cloud hop latency")
    return parser.parse_args()


def build_devices(args: argparse.Namespace) -> List[DeviceSpec]:
    devices: List[DeviceSpec] = []
    devices.append(
        DeviceSpec(
            device_id=0,
            name="cloud",
            layer="cloud",
            compute_capacity=args.cloud_capacity,
            energy_budget=args.cloud_energy_budget,
            trust_level=args.cloud_trust,
        )
    )
    next_id = 1
    edge_ids = []
    for edge_index in range(args.edges):
        edge_id = next_id
        edge_ids.append(edge_id)
        devices.append(
            DeviceSpec(
                device_id=edge_id,
                name=f"edge-{edge_index}",
                layer="edge",
                compute_capacity=args.edge_capacity,
                energy_budget=args.edge_energy_budget,
                trust_level=args.edge_trust,
            )
        )
        next_id += 1
        for device_index in range(args.devices_per_edge):
            devices.append(
                DeviceSpec(
                    device_id=next_id,
                    name=f"device-{edge_index}-{device_index}",
                    layer="device",
                    parent_edge=edge_id,
                    compute_capacity=args.device_capacity,
                    energy_budget=args.device_energy_budget,
                    trust_level=args.device_trust,
                )
            )
            next_id += 1
    return devices


def compute_output_sizes(nodes: List[Dict], edges: List[Dict]) -> Dict[int, float]:
    output_sizes: Dict[int, float] = {node["id"]: node.get("size", 0.0) for node in nodes}
    for edge in edges:
        source_id = edge["sourceId"]
        edge_size = edge.get("size")
        if edge_size is not None:
            output_sizes[source_id] = max(output_sizes.get(source_id, 0.0), float(edge_size))
    return output_sizes


def build_comm_matrix(
    device_count: int,
    devices: List[DeviceSpec],
    args: argparse.Namespace,
) -> Dict[str, List[List[float]]]:
    bandwidth = [[0.0 for _ in range(device_count)] for _ in range(device_count)]
    latency = [[0.0 for _ in range(device_count)] for _ in range(device_count)]

    device_map = {device.device_id: device for device in devices}

    def set_link(a: int, b: int, bw: float, lat: float) -> None:
        bandwidth[a][b] = bw
        bandwidth[b][a] = bw
        latency[a][b] = lat
        latency[b][a] = lat

    for device in devices:
        if device.layer == "edge":
            set_link(0, device.device_id, args.cloud_edge_bandwidth, args.cloud_edge_latency)
        if device.layer == "device" and device.parent_edge is not None:
            set_link(device.device_id, device.parent_edge, args.edge_device_bandwidth, args.edge_device_latency)

    devices_by_edge: Dict[int, List[int]] = {}
    for device in devices:
        if device.layer == "device" and device.parent_edge is not None:
            devices_by_edge.setdefault(device.parent_edge, []).append(device.device_id)
    for edge_id, child_devices in devices_by_edge.items():
        for i, device_a in enumerate(child_devices):
            for device_b in child_devices[i + 1 :]:
                set_link(device_a, device_b, args.device_device_bandwidth, args.device_device_latency)
    return {"bandwidth": bandwidth, "latency": latency, "cloud_multiplier": args.cloud_multiplier}


def convert_graph(args: argparse.Namespace) -> Dict:
    input_data = json.loads(args.input.read_text())
    nodes = input_data["nodes"]
    edges = input_data["edges"]
    devices = build_devices(args)
    output_sizes = compute_output_sizes(nodes, edges)

    device_count = len(devices)
    cloud_index = 0
    edge_indices = [device.device_id for device in devices if device.layer == "edge"]
    device_indices = [device.device_id for device in devices if device.layer == "device"]

    device_scales = {
        cloud_index: args.cloud_scale,
    }
    for edge_id in edge_indices:
        device_scales[edge_id] = args.edge_scale
    for device_id in device_indices:
        device_scales[device_id] = args.device_scale

    device_energy = {
        cloud_index: args.cloud_energy,
    }
    for edge_id in edge_indices:
        device_energy[edge_id] = args.edge_energy
    for device_id in device_indices:
        device_energy[device_id] = args.device_energy

    operators = []
    for node in nodes:
        base_latency = float(node.get(args.base_latency, 0.0))
        compute_times = [base_latency * device_scales[device.device_id] for device in devices]
        energy_costs = [device_energy[device.device_id] for device in devices]
        operators.append(
            {
                "id": node["id"],
                "name": node.get("name", f"op-{node['id']}"),
                "compute_time": compute_times,
                "output_size": output_sizes.get(node["id"], 0.0),
                "energy_cost": energy_costs,
                "trust_requirement": args.trust_default,
            }
        )

    device_entries = []
    for device in devices:
        entry = {
            "id": device.device_id,
            "name": device.name,
            "layer": device.layer,
            "compute_capacity": device.compute_capacity,
            "energy_budget": device.energy_budget,
            "trust_level": device.trust_level,
        }
        if device.parent_edge is not None:
            entry["parent_edge"] = device.parent_edge
        device_entries.append(entry)

    converted_edges = [{"source": edge["sourceId"], "dest": edge["destId"]} for edge in edges]
    communication = build_comm_matrix(len(devices), devices, args)

    return {
        "devices": device_entries,
        "operators": operators,
        "edges": converted_edges,
        "communication": communication,
        "weights": {"alpha": 1.0, "beta": 0.1, "gamma": 0.5},
    }


def main() -> None:
    args = parse_args()
    data = convert_graph(args)
    args.output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
