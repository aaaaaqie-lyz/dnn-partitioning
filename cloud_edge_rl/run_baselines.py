#!/usr/bin/env python3
"""Run baseline placement strategies and output split JSON."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from baseline_solvers import solve_dp, solve_ip
from rl_env import PlacementEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument(
        "--method",
        choices=["random", "greedy", "greedy-aware", "rl", "dp", "ip"],
        default="greedy",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Used for rl method")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Used for rl method")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Used for rl method")
    parser.add_argument("--cloud-penalty", type=float, default=2.0, help="Layer penalty for cloud usage (greedy-aware)")
    parser.add_argument(
        "--privacy-threshold",
        type=float,
        default=0.7,
        help="Trust threshold to avoid cloud in greedy-aware",
    )
    parser.add_argument(
        "--cross-layer-penalty",
        type=float,
        default=2.0,
        help="Penalty for cross-layer communication in greedy-aware",
    )
    parser.add_argument("--output", type=Path, help="Write split JSON to file (defaults to stdout)")
    return parser.parse_args()


def snapshot_env(env: PlacementEnv) -> Dict:
    return {
        "assigned": dict(env.assigned),
        "device_loads": list(env.device_loads),
        "device_compute_loads": list(env.device_compute_loads),
        "device_comm_loads": list(env.device_comm_loads),
        "device_energy": list(env.device_energy),
        "device_trust_penalty": list(env.device_trust_penalty),
    }


def restore_env(env: PlacementEnv, snapshot: Dict) -> None:
    env.assigned = dict(snapshot["assigned"])
    env.device_loads = list(snapshot["device_loads"])
    env.device_compute_loads = list(snapshot["device_compute_loads"])
    env.device_comm_loads = list(snapshot["device_comm_loads"])
    env.device_energy = list(snapshot["device_energy"])
    env.device_trust_penalty = list(snapshot["device_trust_penalty"])


def run_random(env: PlacementEnv) -> None:
    env.reset()
    while not env.is_done():
        frontier = env.frontier()
        if not frontier:
            break
        op_id = random.choice(frontier)
        device_id = random.choice(env.valid_devices(op_id))
        env.step(op_id, device_id)


def run_greedy(env: PlacementEnv) -> None:
    env.reset()
    while not env.is_done():
        frontier = env.frontier()
        if not frontier:
            break
        op_id = min(frontier)
        best_device = None
        best_objective = float("inf")
        for device_id in env.valid_devices(op_id):
            snap = snapshot_env(env)
            env.step(op_id, device_id)
            objective = env.objective()
            if objective < best_objective:
                best_objective = objective
                best_device = device_id
            restore_env(env, snap)
        if best_device is None:
            break
        env.step(op_id, best_device)


def is_cross_layer(env: PlacementEnv, src_device: int, dst_device: int) -> bool:
    return env.devices[src_device].layer != env.devices[dst_device].layer


def run_greedy_aware(env: PlacementEnv, args: argparse.Namespace) -> None:
    env.reset()
    layer_weights = {
        "device": 1.0,
        "edge": 1.2,
        "cloud": args.cloud_penalty,
    }
    while not env.is_done():
        frontier = env.frontier()
        if not frontier:
            break
        op_id = min(frontier)
        operator = env.op_map[op_id]
        best_device = None
        best_score = float("inf")
        valid_devices = env.valid_devices(op_id)
        preferred_devices = valid_devices
        if operator.trust_requirement > args.privacy_threshold:
            preferred_devices = [d for d in valid_devices if env.devices[d].layer != "cloud"]
            if not preferred_devices:
                preferred_devices = valid_devices

        for device_id in preferred_devices:
            snap = snapshot_env(env)
            env.step(op_id, device_id)
            base_objective = env.objective()
            layer_penalty = layer_weights[env.devices[device_id].layer]
            comm_penalty = 0.0
            for pred_id in env.predecessors[op_id]:
                if pred_id in env.assigned:
                    pred_device = env.assigned[pred_id]
                    if is_cross_layer(env, pred_device, device_id):
                        dep_weight = env.dependency_weight(pred_id, op_id)
                        comm_penalty += args.cross_layer_penalty * (1.0 + dep_weight)
            score = base_objective * layer_penalty + comm_penalty
            if score < best_score:
                best_score = score
                best_device = device_id
            restore_env(env, snap)

        if best_device is None:
            break
        env.step(op_id, best_device)


def select_action(
    q_table: Dict[Tuple[int, int], float],
    op_id: int,
    valid_devices: List[int],
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.choice(valid_devices)
    return max(valid_devices, key=lambda d: q_table.get((op_id, d), 0.0))


def run_rl(env: PlacementEnv, args: argparse.Namespace) -> Dict:
    q_table: Dict[Tuple[int, int], float] = {}
    best_output: Dict | None = None
    best_objective = float("inf")

    for _ in range(args.episodes):
        env.reset()
        while not env.is_done():
            frontier = env.frontier()
            if not frontier:
                break
            op_id = random.choice(frontier)
            device_id = select_action(q_table, op_id, env.valid_devices(op_id), args.epsilon)
            reward, _ = env.step(op_id, device_id)
            key = (op_id, device_id)
            old_value = q_table.get(key, 0.0)
            q_table[key] = old_value + args.learning_rate * (reward - old_value)
        if env.is_done() and env.objective() < best_objective:
            best_objective = env.objective()
            best_output = env.split_output()

    if best_output is None:
        best_output = env.split_output()
    return best_output


def main() -> None:
    args = parse_args()
    env = PlacementEnv(args.graph)

    start_time = time.perf_counter()
    if args.method == "random":
        run_random(env)
        output = env.split_output()
    elif args.method == "greedy":
        run_greedy(env)
        output = env.split_output()
    elif args.method == "greedy-aware":
        run_greedy_aware(env, args)
        output = env.split_output()
    elif args.method == "dp":
        order = env.topological_order()
        result = solve_dp(env, order)
        output = {
            "devices": [
                {
                    "id": device.device_id,
                    "layer": device.layer,
                    "compute_load": result.device_compute_loads[device.device_id],
                    "comm_load": result.device_comm_loads[device.device_id],
                    "total_load": result.device_loads[device.device_id],
                    "nodes": sorted(
                        [op_id for op_id, dev_id in result.assignment.items() if dev_id == device.device_id]
                    ),
                }
                for device in env.devices
            ],
            "objective": result.objective,
        }
    elif args.method == "ip":
        order = env.topological_order()
        result = solve_ip(env, order)
        output = {
            "devices": [
                {
                    "id": device.device_id,
                    "layer": device.layer,
                    "compute_load": result.device_compute_loads[device.device_id],
                    "comm_load": result.device_comm_loads[device.device_id],
                    "total_load": result.device_loads[device.device_id],
                    "nodes": sorted(
                        [op_id for op_id, dev_id in result.assignment.items() if dev_id == device.device_id]
                    ),
                }
                for device in env.devices
            ],
            "objective": result.objective,
        }
    else:
        output = run_rl(env, args)
    end_time = time.perf_counter()
    output["method"] = args.method
    output["execution_time_ms"] = (end_time - start_time) * 1000.0

    output_text = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
