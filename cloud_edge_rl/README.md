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
- `run_baselines.py`: Runs baseline placement methods (random / greedy / RL / DP / IP) and emits a split JSON.
- `sample_rl_graph.json`: A tiny example graph in the new JSON format.

## Quick start

### 1) Convert an existing operator graph

```bash
python cloud_edge_rl/convert_paper_json.py \
  --input throughput-inputs/OperatorGraphs/bert_l-3_inference.json \
  --output cloud_edge_rl/bert_l-3_cloud_edge.json \
  --edges 2 \
  --devices-per-edge 2
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

The scripts emit split JSON to match the repo’s original output style (device lists with `load` and `nodes`), and include `execution_time_ms`.

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
      "trust_requirement": 0.6
    }
  ],
  "edges": [
    {"source": 0, "dest": 1}
  ],
  "communication": {
    "bandwidth": [[0, 1000000000, 0], [1000000000, 0, 500000000], [0, 500000000, 0]],
    "latency": [[0, 8, 0], [8, 0, 1], [0, 1, 0]]
  },
  "weights": {"alpha": 1.0, "beta": 0.1, "gamma": 0.5}
}
```

### Notes

- `compute_time` and `energy_cost` are arrays aligned with device IDs.
- Device IDs must be contiguous and match the row/column indices in the communication matrices.
- `communication` defines direct-link latency/bandwidth. Multi-hop routing is computed by `rl_env.py` using the Cloud–Edge–Device constraints, including device-to-device links under the same edge server.
- The objective minimized is the **max** over devices of `alpha * T_d + beta * E_d + gamma * trust_penalty`.

## Split output format (mirrors original style)

```json
{
  "method": "greedy",
  "objective": 12.34,
  "devices": [
    {
      "id": 0,
      "layer": "cloud",
      "load": 3.21,
      "nodes": [0, 3, 7]
    },
    {
      "id": 1,
      "layer": "edge",
      "load": 6.54,
      "nodes": [1, 2]
    }
  ],
  "execution_time_ms": 42.0
}
```
