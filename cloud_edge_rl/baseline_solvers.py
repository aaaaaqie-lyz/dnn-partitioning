#!/usr/bin/env python3
"""Baseline solvers for operator placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rl_env import PlacementEnv


@dataclass
class SolverResult:
    assignment: Dict[int, int]
    device_loads: List[float]
    device_compute_loads: List[float]
    device_comm_loads: List[float]
    device_energy: List[float]
    device_trust_penalty: List[float]
    objective: float


def _objective(weights, loads, energy, trust) -> float:
    per_device = [
        weights.alpha * load + weights.beta * energy + weights.gamma * trust
        for load, energy, trust in zip(loads, energy, trust)
    ]
    return max(per_device) if per_device else 0.0


def _apply_assignment(
    env: PlacementEnv,
    op_id: int,
    device_id: int,
    assignment: Dict[int, int],
    loads: List[float],
    compute_loads: List[float],
    comm_loads: List[float],
    energy: List[float],
    trust_penalty: List[float],
) -> Optional[Tuple[List[float], List[float], List[float], List[float], List[float]]]:
    operator = env.op_map[op_id]
    device = env.devices[device_id]

    compute_time = operator.compute_time[device_id]
    projected_energy = energy[device_id] + compute_time * operator.energy_cost[device_id]
    if projected_energy > device.energy_budget:
        return None

    new_loads = list(loads)
    new_compute = list(compute_loads)
    new_comm = list(comm_loads)
    new_energy = list(energy)
    new_trust = list(trust_penalty)

    new_loads[device_id] += compute_time
    new_compute[device_id] += compute_time
    new_energy[device_id] = projected_energy

    trust_gap = max(0.0, operator.trust_requirement - device.trust_level)
    new_trust[device_id] += trust_gap * trust_gap

    for pred in env.predecessors[op_id]:
        pred_device = assignment[pred]
        if pred_device != device_id:
            pred_output = env.op_map[pred].output_size
            dep_weight = env.dependency_weight(pred, op_id)
            comm_latency = env.scaled_comm_latency(pred_device, device_id, pred_output, dep_weight)
            new_loads[device_id] += comm_latency
            new_comm[device_id] += comm_latency

    return new_loads, new_compute, new_comm, new_energy, new_trust


def solve_dp(env: PlacementEnv, order: List[int]) -> SolverResult:
    cache: Dict[Tuple[int, Tuple[int, ...]], SolverResult] = {}

    def recurse(index: int, assignment: Dict[int, int], loads, compute_loads, comm_loads, energy, trust) -> SolverResult:
        key = (index, tuple(assignment.get(op_id, -1) for op_id in order[:index]))
        if key in cache:
            return cache[key]

        if index == len(order):
            result = SolverResult(
                assignment=dict(assignment),
                device_loads=list(loads),
                device_compute_loads=list(compute_loads),
                device_comm_loads=list(comm_loads),
                device_energy=list(energy),
                device_trust_penalty=list(trust),
                objective=_objective(env.weights, loads, energy, trust),
            )
            cache[key] = result
            return result

        op_id = order[index]
        best_result: Optional[SolverResult] = None
        for device_id in range(len(env.devices)):
            updated = _apply_assignment(env, op_id, device_id, assignment, loads, compute_loads, comm_loads, energy, trust)
            if updated is None:
                continue
            new_loads, new_compute, new_comm, new_energy, new_trust = updated
            assignment[op_id] = device_id
            candidate = recurse(index + 1, assignment, new_loads, new_compute, new_comm, new_energy, new_trust)
            assignment.pop(op_id)
            if best_result is None or candidate.objective < best_result.objective:
                best_result = candidate

        if best_result is None:
            best_result = SolverResult(
                assignment=dict(assignment),
                device_loads=list(loads),
                device_compute_loads=list(compute_loads),
                device_comm_loads=list(comm_loads),
                device_energy=list(energy),
                device_trust_penalty=list(trust),
                objective=_objective(env.weights, loads, energy, trust),
            )
        cache[key] = best_result
        return best_result

    return recurse(
        0,
        {},
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
    )


def solve_ip(env: PlacementEnv, order: List[int]) -> SolverResult:
    best_result: Optional[SolverResult] = None

    def recurse(index: int, assignment: Dict[int, int], loads, compute_loads, comm_loads, energy, trust) -> None:
        nonlocal best_result
        current_obj = _objective(env.weights, loads, energy, trust)
        if best_result is not None and current_obj >= best_result.objective:
            return
        if index == len(order):
            candidate = SolverResult(
                assignment=dict(assignment),
                device_loads=list(loads),
                device_compute_loads=list(compute_loads),
                device_comm_loads=list(comm_loads),
                device_energy=list(energy),
                device_trust_penalty=list(trust),
                objective=current_obj,
            )
            if best_result is None or candidate.objective < best_result.objective:
                best_result = candidate
            return

        op_id = order[index]
        for device_id in range(len(env.devices)):
            updated = _apply_assignment(env, op_id, device_id, assignment, loads, compute_loads, comm_loads, energy, trust)
            if updated is None:
                continue
            new_loads, new_compute, new_comm, new_energy, new_trust = updated
            assignment[op_id] = device_id
            recurse(index + 1, assignment, new_loads, new_compute, new_comm, new_energy, new_trust)
            assignment.pop(op_id)

    recurse(
        0,
        {},
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
        [0.0 for _ in env.devices],
    )

    if best_result is None:
        best_result = SolverResult(
            assignment={},
            device_loads=[0.0 for _ in env.devices],
            device_compute_loads=[0.0 for _ in env.devices],
            device_comm_loads=[0.0 for _ in env.devices],
            device_energy=[0.0 for _ in env.devices],
            device_trust_penalty=[0.0 for _ in env.devices],
            objective=0.0,
        )
    return best_result
