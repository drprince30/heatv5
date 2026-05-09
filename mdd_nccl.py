
from __future__ import annotations
import pandas as pd
from mdd_core import robust_zscore, sigmoid, component_id

def analyze_nccl(nccl: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "job_id", "rank", "node_id", "gpu_id", "latency_ms", "unresponsive", "timeout_count"}
    missing = required - set(nccl.columns)
    if missing:
        raise ValueError(f"nccl data missing columns: {sorted(missing)}")
    n = nccl.copy()
    n["node_id"] = n["node_id"].astype(str)
    n["gpu_id"] = n["gpu_id"].astype(str)
    n["component_id"] = [component_id(a, b) for a, b in zip(n["node_id"], n["gpu_id"])]
    for c in ["rank", "latency_ms", "unresponsive", "timeout_count"]:
        n[c] = pd.to_numeric(n[c], errors="coerce").fillna(0)
    rows = []
    for job, gj in n.groupby("job_id", sort=False):
        gj = gj.copy()
        gj["latency_z"] = robust_zscore(gj["latency_ms"]).values
        for _, row in gj.iterrows():
            latency_score = sigmoid(float(row["latency_z"]) - 1.8)
            unresp = 1.0 if float(row["unresponsive"]) > 0 else 0.0
            timeout = min(1.0, float(row["timeout_count"]) / 3.0)
            ras_state = str(row.get("ras_state", "OK")).upper()
            ras_score = 1.0 if "UNRESPONSIVE" in ras_state else 0.75 if "WARN" in ras_state else 0.55 if "SLOW" in ras_state else 0.0
            nccl_score = max(0, min(1, 0.35 * latency_score + 0.25 * unresp + 0.20 * timeout + 0.20 * ras_score))
            rows.append({
                "timestamp": row["timestamp"], "job_id": row["job_id"], "rank": int(row["rank"]),
                "node_id": row["node_id"], "gpu_id": row["gpu_id"], "component_id": row["component_id"],
                "latency_ms": float(row["latency_ms"]), "latency_z": float(row["latency_z"]),
                "unresponsive": unresp, "timeout_score": timeout, "ras_state": ras_state,
                "nccl_score": float(nccl_score),
                "nccl_warning": "NCCL_RING_STALL_RISK" if nccl_score >= 0.65 else "ok",
            })
    return pd.DataFrame(rows)

def job_nccl_summary(nccl_score: pd.DataFrame) -> pd.DataFrame:
    if nccl_score.empty:
        return pd.DataFrame()
    return nccl_score.groupby("job_id").agg(
        peak_nccl_score=("nccl_score", "max"),
        slow_ranks=("nccl_warning", lambda x: int((x != "ok").sum())),
        peak_latency=("latency_ms", "max"),
    ).reset_index().sort_values("peak_nccl_score", ascending=False)
