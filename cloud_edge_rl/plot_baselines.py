#!/usr/bin/env python3
"""Plot baseline comparison charts from split JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Split JSON files")
    parser.add_argument("--output", type=Path, default=Path("baseline_comparison.png"))
    return parser.parse_args()


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())


def max_total_load(devices: List[dict]) -> float:
    return max(device.get("total_load", 0.0) for device in devices) if devices else 0.0


def main() -> None:
    args = parse_args()
    splits = [load_split(path) for path in args.inputs]

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it or skip plotting.") from exc

    labels = [split.get("method", path.stem) for split, path in zip(splits, args.inputs)]
    objectives = [split.get("objective", 0.0) for split in splits]
    max_loads = [max_total_load(split.get("devices", [])) for split in splits]

    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x, objectives, color="#4C72B0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_title("Max objective")

    axes[1].bar(x, max_loads, color="#55A868")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_title("Max total load")

    fig.tight_layout()
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
