#!/usr/bin/env python3
"""Minimal Q-learning baseline for Cloud–Edge–Device placement."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent))

from rl_env import PlacementEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    return parser.parse_args()


def select_action(
    q_table: Dict[Tuple[int, int], float],
    op_id: int,
    valid_devices: List[int],
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        return random.choice(valid_devices)
    best_device = max(valid_devices, key=lambda d: q_table.get((op_id, d), 0.0))
    return best_device


def run_episode(env: PlacementEnv, q_table: Dict[Tuple[int, int], float], args: argparse.Namespace) -> float:
    env.reset()
    total_reward = 0.0
    while not env.is_done():
        frontier = env.frontier()
        if not frontier:
            break
        op_id = random.choice(frontier)
        device_id = select_action(q_table, op_id, env.valid_devices(op_id), args.epsilon)
        reward, _ = env.step(op_id, device_id)
        total_reward += reward
        key = (op_id, device_id)
        old_value = q_table.get(key, 0.0)
        q_table[key] = old_value + args.learning_rate * (reward - old_value)
    return total_reward


def main() -> None:
    args = parse_args()
    env = PlacementEnv(args.graph)
    q_table: Dict[Tuple[int, int], float] = {}
    best_objective = float("inf")
    best_assignment = None

    for _ in range(args.episodes):
        run_episode(env, q_table, args)
        summary = env.summary()
        if summary["objective"] < best_objective and env.is_done():
            best_objective = summary["objective"]
            best_assignment = summary

    print("Best objective:", best_objective)
    if best_assignment:
        print("Assignment:", best_assignment["assigned"])
        print("Device loads:", best_assignment["device_loads"])
        print("Device energy:", best_assignment["device_energy"])
        print("Device trust penalty:", best_assignment["device_trust_penalty"])


if __name__ == "__main__":
    main()
