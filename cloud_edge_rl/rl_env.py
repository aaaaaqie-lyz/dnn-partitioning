#!/usr/bin/env python3
"""RL environment for Cloud–Edge–Device operator placement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Device:
    device_id: int
    name: str
    layer: str
    compute_capacity: float
    energy_budget: float
    trust_level: float
    parent_edge: Optional[int] = None


@dataclass
class Operator:
    op_id: int
    name: str
    compute_time: List[float]
    output_size: float
    energy_cost: List[float]
    trust_requirement: float


@dataclass
class Weights:
    alpha: float
    beta: float
    gamma: float


class PlacementEnv:
    def __init__(self, graph_path: Path) -> None:
        data = json.loads(graph_path.read_text())
        self.devices = [
            Device(
                device_id=device["id"],
                name=device["name"],
                layer=device["layer"],
                compute_capacity=device["compute_capacity"],
                energy_budget=device["energy_budget"],
                trust_level=device["trust_level"],
                parent_edge=device.get("parent_edge"),
            )
            for device in data["devices"]
        ]
        self.operators = [
            Operator(
                op_id=op["id"],
                name=op.get("name", f"op-{op['id']}"),
                compute_time=op["compute_time"],
                output_size=op["output_size"],
                energy_cost=op["energy_cost"],
                trust_requirement=op["trust_requirement"],
            )
            for op in data["operators"]
        ]
        self.edges = [(edge["source"], edge["dest"]) for edge in data["edges"]]
        weights = data.get("weights", {"alpha": 1.0, "beta": 0.1, "gamma": 0.5})
        self.weights = Weights(**weights)
        self.bandwidth = data["communication"]["bandwidth"]
        self.latency = data["communication"]["latency"]

        self.op_map = {op.op_id: op for op in self.operators}
        self.successors: Dict[int, List[int]] = {op.op_id: [] for op in self.operators}
        self.predecessors: Dict[int, List[int]] = {op.op_id: [] for op in self.operators}
        for src, dst in self.edges:
            self.successors[src].append(dst)
            self.predecessors[dst].append(src)

        self.reset()

    def reset(self) -> None:
        self.assigned: Dict[int, int] = {}
        self.device_loads = [0.0 for _ in self.devices]
        self.device_energy = [0.0 for _ in self.devices]
        self.device_trust_penalty = [0.0 for _ in self.devices]

    def frontier(self) -> List[int]:
        available = []
        for op_id in self.op_map:
            if op_id in self.assigned:
                continue
            if all(pred in self.assigned for pred in self.predecessors[op_id]):
                available.append(op_id)
        return available

    def is_done(self) -> bool:
        return len(self.assigned) == len(self.op_map)

    def _direct_comm(self, src: int, dst: int, output_size: float) -> Optional[float]:
        bw = self.bandwidth[src][dst]
        if bw <= 0:
            return None
        return self.latency[src][dst] + output_size / bw

    def _find_device(self, device_id: int) -> Device:
        return next(device for device in self.devices if device.device_id == device_id)

    def communication_latency(self, src: int, dst: int, output_size: float) -> float:
        if src == dst:
            return 0.0
        direct = self._direct_comm(src, dst, output_size)
        if direct is not None:
            return direct

        src_device = self._find_device(src)
        dst_device = self._find_device(dst)

        def hop_latency(a: int, b: int) -> float:
            direct_latency = self._direct_comm(a, b, output_size)
            if direct_latency is None:
                raise ValueError(f"No communication path between {a} and {b}")
            return direct_latency

        if src_device.layer == "edge" and dst_device.layer == "edge":
            return hop_latency(src, 0) + hop_latency(0, dst)

        if src_device.layer == "device" and dst_device.layer == "device":
            if src_device.parent_edge == dst_device.parent_edge:
                return hop_latency(src, dst)
            return (
                hop_latency(src, src_device.parent_edge)
                + hop_latency(src_device.parent_edge, 0)
                + hop_latency(0, dst_device.parent_edge)
                + hop_latency(dst_device.parent_edge, dst)
            )

        if src_device.layer == "device" and dst_device.layer == "edge":
            if src_device.parent_edge == dst:
                return hop_latency(src, dst)
            return hop_latency(src, src_device.parent_edge) + hop_latency(src_device.parent_edge, 0) + hop_latency(0, dst)

        if src_device.layer == "edge" and dst_device.layer == "device":
            if dst_device.parent_edge == src:
                return hop_latency(src, dst)
            return hop_latency(src, 0) + hop_latency(0, dst_device.parent_edge) + hop_latency(dst_device.parent_edge, dst)

        if src_device.layer == "device" and dst_device.layer == "cloud":
            return hop_latency(src, src_device.parent_edge) + hop_latency(src_device.parent_edge, 0)

        if src_device.layer == "cloud" and dst_device.layer == "device":
            return hop_latency(0, dst_device.parent_edge) + hop_latency(dst_device.parent_edge, dst)

        raise ValueError(f"Unsupported communication route from {src} to {dst}")

    def objective(self) -> float:
        per_device = [
            self.weights.alpha * load + self.weights.beta * energy + self.weights.gamma * trust
            for load, energy, trust in zip(self.device_loads, self.device_energy, self.device_trust_penalty)
        ]
        return max(per_device) if per_device else 0.0

    def step(self, op_id: int, device_id: int) -> Tuple[float, bool]:
        if op_id not in self.frontier():
            return -10.0, False
        if device_id < 0 or device_id >= len(self.devices):
            return -10.0, False

        operator = self.op_map[op_id]
        device = self.devices[device_id]
        if device.energy_budget < self.device_energy[device_id]:
            return -5.0, False

        before_obj = self.objective()
        compute_time = operator.compute_time[device_id]
        self.device_loads[device_id] += compute_time
        self.device_energy[device_id] += compute_time * operator.energy_cost[device_id]

        trust_gap = max(0.0, operator.trust_requirement - device.trust_level)
        self.device_trust_penalty[device_id] += trust_gap * trust_gap

        for pred in self.predecessors[op_id]:
            pred_device = self.assigned[pred]
            if pred_device != device_id:
                pred_output = self.op_map[pred].output_size
                comm_latency = self.communication_latency(pred_device, device_id, pred_output)
                self.device_loads[device_id] += comm_latency

        self.assigned[op_id] = device_id
        after_obj = self.objective()
        reward = -(after_obj - before_obj)
        return reward, self.is_done()

    def valid_devices(self, op_id: int) -> List[int]:
        return list(range(len(self.devices)))

    def summary(self) -> Dict:
        return {
            "assigned": self.assigned,
            "device_loads": self.device_loads,
            "device_energy": self.device_energy,
            "device_trust_penalty": self.device_trust_penalty,
            "objective": self.objective(),
        }

    def split_output(self) -> Dict:
        device_nodes: Dict[int, List[int]] = {device.device_id: [] for device in self.devices}
        for op_id, device_id in self.assigned.items():
            device_nodes[device_id].append(op_id)
        for node_list in device_nodes.values():
            node_list.sort()
        device_entries = []
        for device in self.devices:
            device_entries.append(
                {
                    "id": device.device_id,
                    "layer": device.layer,
                    "load": self.device_loads[device.device_id],
                    "nodes": device_nodes[device.device_id],
                }
            )
        return {
            "devices": device_entries,
            "objective": self.objective(),
        }
