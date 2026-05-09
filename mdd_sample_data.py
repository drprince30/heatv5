
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

SCENARIOS = [
    "normal_training",
    "partial_metric_collapse",
    "xid79_detachment",
    "nccl_ring_stall",
    "fabric_retry_storm",
    "pcie_degradation",
    "nvlink_gray_failure",
    "checkpoint_false_positive",
    "stale_scrape_hidden_failure",
]

def _timestamp(t: int) -> str:
    return f"t{t:04d}"

def generate_dataset(scenario: str = "partial_metric_collapse", steps: int = 160, nodes: int = 12, gpus_per_node: int = 4, seed: int = 17) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    topology_rows, overlay_rows = [], []
    job_id = "train-job-001"
    rank = 0
    for n in range(nodes):
        node = f"gpu-node-{n:02d}"
        rack = f"rack-{n // 4 + 1}"
        fabric = f"fabric-{(n % 4) // 2}"
        switch = f"leaf-{n // 4}-{(n % 4) // 2}"
        for g in range(gpus_per_node):
            topology_rows.append({
                "node_id": node, "gpu_id": g, "rack_id": rack, "fabric_group": fabric,
                "switch_id": switch, "pci_bus": f"0000:{20+n:02x}:{g}.0",
                "power_domain": f"pdu-{n // 4 + 1}", "cooling_loop": f"loop-{1 if n % 4 < 2 else 2}",
            })
            overlay_rows.append({
                "job_id": job_id, "rank": rank, "node_id": node, "gpu_id": g,
                "fabric_group": fabric, "switch_id": switch, "training_role": "tp-rank" if rank % 2 == 0 else "dp-rank",
                "tenant_id": f"tenant-{n % 3}", "rank_importance": 1.0 + (0.25 if rank % 8 == 0 else 0.0),
            })
            rank += 1

    telemetry_rows, scrape_rows, nccl_rows = [], [], []
    target_node, target_gpu = "gpu-node-05", 2
    fabric_nodes = {"gpu-node-04", "gpu-node-05", "gpu-node-06", "gpu-node-07"}

    for t in range(steps):
        progress = t / max(1, steps - 1)
        for n in range(nodes):
            node = f"gpu-node-{n:02d}"
            target = f"dcgm-{node}:9400"
            base_samples = 820 + int(rng.normal(0, 12))
            base_families = 74 + int(rng.normal(0, 2))
            duration = max(0.05, 0.42 + rng.normal(0, 0.04))
            up = 1
            gpu_metric_count = 96 + int(rng.normal(0, 3))

            if scenario == "partial_metric_collapse" and node == target_node and progress > 0.55:
                gpu_metric_count = max(5, int(96 * (1 - 0.70 * (progress - 0.55) / 0.45)))
                base_samples = max(120, int(base_samples * (1 - 0.55 * (progress - 0.55) / 0.45)))
                duration = 1.6 + 2.0 * progress
            elif scenario == "xid79_detachment" and node == target_node and progress > 0.62:
                gpu_metric_count = max(0, int(96 * (1 - 1.2 * (progress - 0.62) / 0.38)))
                base_samples = max(30, int(base_samples * (1 - 0.90 * (progress - 0.62) / 0.38)))
                duration = 2.4 + 2.0 * progress
            elif scenario == "stale_scrape_hidden_failure" and node == target_node and progress > 0.50:
                duration, base_samples, gpu_metric_count = 0.40, 820, 96
            elif scenario == "checkpoint_false_positive" and 0.45 < progress < 0.52:
                duration = 1.2
                base_samples = int(base_samples * 0.92)

            scrape_rows.append({
                "timestamp": _timestamp(t), "target": target, "node_id": node, "up": up,
                "scrape_duration_s": round(float(duration), 3),
                "scrape_samples_scraped": int(base_samples),
                "dcgm_gpu_metric_count": int(gpu_metric_count),
                "dcgm_metric_families_count": int(base_families),
            })

            for g in range(gpus_per_node):
                base_util = 72 + rng.normal(0, 8)
                power = 250 + base_util * 1.2 + rng.normal(0, 18)
                temp = 62 + power * 0.04 + rng.normal(0, 2.5)
                pcie_replay = nvlink_crc = ib_crc = packet_drops = ecc = xid = 0
                nccl_latency = 12 + rng.normal(0, 2)
                ras_state, unresponsive, timeout = "OK", 0, 0

                if scenario == "fabric_retry_storm" and node in fabric_nodes and progress > 0.45:
                    ib_crc = int(80 * (progress - 0.45)); packet_drops = int(120 * (progress - 0.45))
                    nccl_latency += 80 * (progress - 0.45); timeout = int(progress > 0.70)
                    ras_state = "SLOW" if progress < 0.75 else "WARN"
                if scenario == "pcie_degradation" and node == target_node and progress > 0.45:
                    pcie_replay = int(150 * (progress - 0.45)); nccl_latency += 35 * (progress - 0.45)
                if scenario == "nvlink_gray_failure" and node in {"gpu-node-06", "gpu-node-07"} and progress > 0.48:
                    nvlink_crc = int(100 * (progress - 0.48)); nccl_latency += 65 * (progress - 0.48)
                if scenario == "nccl_ring_stall" and node == target_node and g == target_gpu and progress > 0.50:
                    nccl_latency += 180 * (progress - 0.50); unresponsive = int(progress > 0.72)
                    timeout = int(progress > 0.78); ras_state = "UNRESPONSIVE" if progress > 0.72 else "SLOW"
                if scenario == "xid79_detachment" and node == target_node and g == target_gpu and progress > 0.72:
                    xid = 79
                if scenario == "partial_metric_collapse" and node == target_node and g == target_gpu and progress > 0.70:
                    pcie_replay = int(80 * (progress - 0.70))
                if scenario == "stale_scrape_hidden_failure" and node == target_node and g == target_gpu and progress > 0.60:
                    pcie_replay = int(100 * (progress - 0.60)); temp = 62.0
                if scenario == "checkpoint_false_positive" and 0.45 < progress < 0.52:
                    nccl_latency += 18; base_util = 15; power = 120

                throttle = int(temp > 91 or xid == 79 or timeout > 0)
                telemetry_rows.append({
                    "timestamp": _timestamp(t), "node_id": node, "gpu_id": g,
                    "temp_c": round(float(temp), 2), "power_w": round(float(power), 2),
                    "gpu_util": round(float(np.clip(base_util, 0, 100)), 2),
                    "ecc_errors": ecc, "xid_errors": xid, "pcie_replay": pcie_replay,
                    "nvlink_crc": nvlink_crc, "ib_crc": ib_crc, "packet_drops": packet_drops,
                    "nccl_latency_ms": round(float(nccl_latency), 2), "throttle_flag": throttle,
                    "job_id": job_id,
                })
                overlay = overlay_rows[n * gpus_per_node + g]
                nccl_rows.append({
                    "timestamp": _timestamp(t), "job_id": job_id, "comm_id": "comm-0",
                    "rank": overlay["rank"], "node_id": node, "gpu_id": g,
                    "ras_state": ras_state, "latency_ms": round(float(nccl_latency), 2),
                    "unresponsive": unresponsive, "timeout_count": timeout,
                })

    return {
        "telemetry": pd.DataFrame(telemetry_rows),
        "scrape": pd.DataFrame(scrape_rows),
        "nccl": pd.DataFrame(nccl_rows),
        "overlay": pd.DataFrame(overlay_rows),
        "topology": pd.DataFrame(topology_rows),
    }

def save_sample_files(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    for name, df in data.items():
        df.to_csv(output_dir / f"sample_{name}.csv", index=False)
