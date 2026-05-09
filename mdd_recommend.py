
from __future__ import annotations
import pandas as pd
from mdd_core import MDDConfig

def build_recommendations(mvi: pd.DataFrame, config: MDDConfig = MDDConfig()) -> pd.DataFrame:
    if mvi.empty:
        return pd.DataFrame(columns=["priority", "target", "class", "mvi_score", "evidence", "recommendation", "command_hint"])
    risky = mvi[mvi["mvi_score"] >= min(0.50, config.mvi_trigger)].copy()
    if risky.empty:
        return pd.DataFrame([{"priority": "info", "target": "cluster", "class": "OK", "mvi_score": 0, "evidence": "No high MVI components.", "recommendation": "Continue shadow monitoring.", "command_hint": ""}])
    rows = []
    top = risky.sort_values("mvi_score", ascending=False).head(25)
    for _, r in top.iterrows():
        cls = r["mdd_class"]
        priority = "critical" if r["mvi_score"] >= 0.78 else "high" if r["mvi_score"] >= 0.64 else "warning"
        target = f"{r['node_id']} GPU {r['gpu_id']}"
        if cls == "SILENT_DETACHMENT_OR_XID79_PATH":
            rec = "Checkpoint affected job if possible; mark node suspect; avoid new ranks; run deep DCGM diagnostic; inspect PCIe bus and driver logs."
            cmd = "dcgmi diag -r 3"
        elif cls == "NCCL_RING_STALL_RISK":
            rec = "Check NCCL RAS for unresponsive rank; checkpoint before watchdog; rebuild communicator or avoid suspect rank/node."
            cmd = "ncclras --query all  # plus scheduler-specific checkpoint command"
        elif cls == "FABRIC_GRAY_FAILURE":
            rec = "Inspect PCIe/NVLink/IB counters; isolate degraded link/switch port; avoid placing collective-heavy jobs on this path."
            cmd = "dcgmi diag -r 3"
        elif cls == "ISOLATED_HARDWARE_FAILURE":
            rec = "Treat as point failure, not cascade. Follow GPU health/RMA workflow and suppress cascade alert unless topology evidence appears."
            cmd = "dcgmi diag -r 2"
        elif cls == "TELEMETRY_COLLAPSE":
            rec = "Restart/check DCGM exporter and verify GPU visibility; do not assume missing metrics mean healthy node."
            cmd = "kubectl rollout restart daemonset/dcgm-exporter -n gpu-operator"
        else:
            rec = "Watch component and verify telemetry/fabric health before disruptive action."
            cmd = ""
        rows.append({"priority": priority, "target": target, "class": cls, "mvi_score": round(float(r["mvi_score"]), 3), "evidence": r.get("evidence", ""), "recommendation": rec, "command_hint": cmd})
    return pd.DataFrame(rows)
