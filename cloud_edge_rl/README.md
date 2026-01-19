# Cloud–Edge–Device RL Placement

This folder adds a lightweight reference pipeline for **operator-level graph conversion** and **reinforcement-learning-based placement** in a three-tier Cloud–Edge–Device system. It is designed to mirror the repo’s JSON operator-graph workflow while adding:

1. **Graph conversion** from the existing paper JSON format into a JSON schema tailored for Cloud–Edge–Device placement.
2. **A minimal RL environment and trainer** that assigns operators to devices under communication, energy, and trust constraints.

> ⚠️ This is a baseline reference implementation (pure Python, no deep learning frameworks). It is intentionally simple and meant to be extended.

## Files

- `convert_paper_json.py`: Converts the paper’s JSON format into a three-tier operator-graph JSON.
- `rl_env.py`: Environment with constraints, communication routing, and min–max objective.
- `train_rl.py`: Minimal Q-learning baseline that demonstrates RL-based placement.
- `baseline_solvers.py`: DP/IP solvers for small graphs (min–max objective, same constraints).
- `run_baselines.py`: Runs baseline placement methods (random / greedy / greedy-aware / RL / DP / IP) and emits a split JSON.
- `plot_baselines.py`: Plots objective and max-load comparisons from split JSON outputs.
- `sample_rl_graph.json`: A tiny example graph in the new JSON format.

## Quick start

### 1) Convert an existing operator graph

```bash
python cloud_edge_rl/convert_paper_json.py \
  --input throughput-inputs/OperatorGraphs/bert_l-3_inference.json \
  --output cloud_edge_rl/bert_l-3_cloud_edge.json \
  --edges 2 \
  --devices-per-edge 2 \
  --cloud-multiplier 2.0 \
  --dependency-weight-scale 1e-9 \
  --criticality-default 1.0
```

### 2) Train a simple RL policy on the converted graph

```bash
python cloud_edge_rl/train_rl.py \
  --graph cloud_edge_rl/bert_l-3_cloud_edge.json \
  --episodes 50 \
  --output cloud_edge_rl/bert_l-3_cloud_edge_split.json
```

### 3) Compare with baseline methods

```bash
python cloud_edge_rl/run_baselines.py \
  --graph cloud_edge_rl/bert_l-3_cloud_edge.json \
  --method greedy \
  --output cloud_edge_rl/bert_l-3_cloud_edge_greedy.json

python cloud_edge_rl/run_baselines.py \
  --graph cloud_edge_rl/bert_l-3_cloud_edge.json \
  --method greedy-aware \
  --cloud-penalty 2.0 \
  --privacy-threshold 0.7 \
  --cross-layer-penalty 2.0 \
  --output cloud_edge_rl/bert_l-3_cloud_edge_greedy_aware.json

python cloud_edge_rl/run_baselines.py \
  --graph cloud_edge_rl/bert_l-3_cloud_edge.json \
  --method random \
  --output cloud_edge_rl/bert_l-3_cloud_edge_random.json

python cloud_edge_rl/run_baselines.py \
  --graph cloud_edge_rl/bert_l-3_cloud_edge.json \
  --method dp \
  --output cloud_edge_rl/bert_l-3_cloud_edge_dp.json

python cloud_edge_rl/run_baselines.py \
  --graph cloud_edge_rl/bert_l-3_cloud_edge.json \
  --method ip \
  --output cloud_edge_rl/bert_l-3_cloud_edge_ip.json
```

The scripts emit split JSON to match the repo’s original output style (device lists with per-device loads and `nodes`), and include `execution_time_ms`.

### 4) Plot baseline comparisons

Requires `matplotlib` to be installed.

Run the baseline scripts first to generate the split JSON files referenced below.

```bash
python cloud_edge_rl/plot_baselines.py \
  --inputs cloud_edge_rl/bert_l-3_cloud_edge_greedy.json \
           cloud_edge_rl/bert_l-3_cloud_edge_greedy_aware.json \
           cloud_edge_rl/bert_l-3_cloud_edge_dp.json \
           cloud_edge_rl/bert_l-3_cloud_edge_ip.json \
  --output cloud_edge_rl/baseline_comparison.png
```

## Cloud-avoidance strategy notes

- `greedy-aware` adds layer preferences (device > edge > cloud), privacy-aware filtering (avoid cloud for high-trust operators), and cross-layer penalties when evaluating a candidate placement.
- RL training can be configured to discourage cloud usage by adjusting `--cloud-penalty`, while `--trust-reward-weight` and `--cross-layer-penalty` weight privacy and cross-layer communication penalties in the reward.
- Dependency-aware placement inflates communication costs by `dependency_weight(u→v) = output_size(u) × criticality(v)` so strong dependencies favor colocation.

## JSON schema (Cloud–Edge–Device)

```json
{
  "devices": [
    {
      "id": 0,
      "name": "cloud",
      "layer": "cloud",
      "compute_capacity": 1000,
      "energy_budget": 100000,
      "trust_level": 0.95
    },
    {
      "id": 1,
      "name": "edge-0",
      "layer": "edge",
      "compute_capacity": 400,
      "energy_budget": 30000,
      "trust_level": 0.85
    },
    {
      "id": 2,
      "name": "device-0-0",
      "layer": "device",
      "parent_edge": 1,
      "compute_capacity": 100,
      "energy_budget": 5000,
      "trust_level": 0.7
    }
  ],
  "operators": [
    {
      "id": 0,
      "name": "MatMul",
      "compute_time": [2.0, 4.0, 8.0],
      "output_size": 4096,
      "energy_cost": [0.2, 0.4, 0.8],
      "trust_requirement": 0.6,
      "criticality": 1.0
    }
  ],
  "edges": [
    {"source": 0, "dest": 1}
  ],
  "communication": {
    "bandwidth": [[0, 1000000000, 0], [1000000000, 0, 500000000], [0, 500000000, 0]],
    "latency": [[0, 8, 0], [8, 0, 1], [0, 1, 0]],
    "cloud_multiplier": 2.0,
    "dependency_weight_scale": 1e-9
  },
  "weights": {"alpha": 1.0, "beta": 0.1, "gamma": 0.5}
}
```

### Notes

- `compute_time` and `energy_cost` are arrays aligned with device IDs.
- Device IDs must be contiguous and match the row/column indices in the communication matrices.
- `communication` defines direct-link latency/bandwidth. Multi-hop routing is computed by `rl_env.py` using the Cloud–Edge–Device constraints, including device-to-device links under the same edge server. Set `cloud_multiplier` to inflate hops that traverse cloud.
- `criticality` scales dependency weight: `dependency_weight(u→v) = output_size(u) × criticality(v)`. `dependency_weight_scale` controls how strongly that weight inflates communication cost.
- The objective minimized is the **max** over devices of `alpha * T_d + beta * E_d + gamma * trust_penalty`.

## Split output format (mirrors original style)

Each device reports compute, communication, and total load so you can see cloud-avoidance effects explicitly.

```json
{
  "method": "greedy",
  "objective": 12.34,
  "devices": [
    {
      "id": 0,
      "layer": "cloud",
      "compute_load": 2.0,
      "comm_load": 1.21,
      "total_load": 3.21,
      "nodes": [0, 3, 7]
    },
    {
      "id": 1,
      "layer": "edge",
      "compute_load": 5.0,
      "comm_load": 1.54,
      "total_load": 6.54,
      "nodes": [1, 2]
    }
  ],
  "execution_time_ms": 42.0
}
```
