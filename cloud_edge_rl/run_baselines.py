#!/usr/bin/env python3
"""Run baseline placement strategies and output split JSON."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from rl_env import PlacementEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--method", choices=["random", "greedy", "rl"], default="greedy")
    parser.add_argument("--episodes", type=int, default=100, help="Used for rl method")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Used for rl method")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Used for rl method")
    parser.add_argument("--output", type=Path, help="Write split JSON to file (defaults to stdout)")
    return parser.parse_args()


def snapshot_env(env: PlacementEnv) -> Dict:
    return {
        "assigned": dict(env.assigned),
        "device_loads": list(env.device_loads),
        "device_energy": list(env.device_energy),
        "device_trust_penalty": list(env.device_trust_penalty),
    }


def restore_env(env: PlacementEnv, snapshot: Dict) -> None:
    env.assigned = dict(snapshot["assigned"])
    env.device_loads = list(snapshot["device_loads"])
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

    if args.method == "random":
        run_random(env)
        output = env.split_output()
    elif args.method == "greedy":
        run_greedy(env)
        output = env.split_output()
    else:
        output = run_rl(env, args)
    output["method"] = args.method

    output_text = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
