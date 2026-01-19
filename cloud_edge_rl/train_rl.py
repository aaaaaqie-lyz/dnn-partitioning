#!/usr/bin/env python3
"""Minimal Q-learning baseline for Cloud–Edge–Device placement."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent))

from rl_env import PlacementEnv, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--cloud-penalty", type=float, default=5.0, help="Penalty for cloud placement")
    parser.add_argument("--trust-reward-weight", type=float, default=1.0, help="Reward weight for trust gap")
    parser.add_argument(
        "--cross-layer-penalty",
        type=float,
        default=2.0,
        help="Penalty for cross-layer communication",
    )
    parser.add_argument("--output", type=Path, help="Write split JSON to file (defaults to stdout)")
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


def run_episode(
    env: PlacementEnv,
    q_table: Dict[Tuple[int, int], float],
    args: argparse.Namespace,
    reward_config: RewardConfig,
) -> float:
    env.reset()
    total_reward = 0.0
    while not env.is_done():
        frontier = env.frontier()
        if not frontier:
            break
        op_id = random.choice(frontier)
        device_id = select_action(q_table, op_id, env.valid_devices(op_id), args.epsilon)
        reward, _ = env.step_with_reward(op_id, device_id, reward_config)
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
    start_time = time.perf_counter()
    reward_config = RewardConfig(
        cloud_penalty=args.cloud_penalty,
        trust_reward_weight=args.trust_reward_weight,
        cross_layer_penalty=args.cross_layer_penalty,
    )

    for _ in range(args.episodes):
        run_episode(env, q_table, args, reward_config)
        if env.is_done() and env.objective() < best_objective:
            best_objective = env.objective()
            best_assignment = env.split_output()

    if best_assignment is None:
        best_assignment = env.split_output()

    best_assignment["method"] = "rl"
    best_assignment["execution_time_ms"] = (time.perf_counter() - start_time) * 1000.0
    output_text = json.dumps(best_assignment, indent=2)
    if args.output:
        args.output.write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
