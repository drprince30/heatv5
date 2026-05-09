
from __future__ import annotations
import pandas as pd
from mdd_core import MDDConfig, MVIWeights, noisy_or, component_id
from mdd_overlay import prepare_overlay

def compute_mvi(telemetry, mata, fabric, nccl, overlay, config=MDDConfig(), weights=MVIWeights()) -> pd.DataFrame:
    t = telemetry.copy()
    t["node_id"] = t["node_id"].astype(str)
    t["gpu_id"] = t["gpu_id"].astype(str)
    t["component_id"] = [component_id(a, b) for a, b in zip(t["node_id"], t["gpu_id"])]
    o = prepare_overlay(overlay)
    m = mata[["timestamp", "node_id", "missingness_score", "mata_warning", "gpu_metric_drop_ratio", "partial_payload"]].copy()
    f = fabric[["timestamp", "node_id", "gpu_id", "component_id", "fabric_score", "fabric_warning", "xid_score", "ecc_score"]].copy()
    n = nccl[["timestamp", "node_id", "gpu_id", "component_id", "job_id", "rank", "nccl_score", "nccl_warning", "latency_ms", "ras_state"]].copy()
    out = t.merge(m, on=["timestamp", "node_id"], how="left")
    out = out.merge(f, on=["timestamp", "node_id", "gpu_id", "component_id"], how="left")
    out = out.merge(n, on=["timestamp", "node_id", "gpu_id", "component_id"], how="left", suffixes=("", "_nccl"))
    out = out.merge(o[["job_id", "rank", "component_id", "fabric_group", "switch_id", "rank_importance", "topology_criticality"]], on=["job_id", "component_id"], how="left", suffixes=("", "_overlay"))
    for c in ["missingness_score", "fabric_score", "xid_score", "ecc_score", "nccl_score", "topology_criticality", "partial_payload", "gpu_metric_drop_ratio"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["detachment_score"] = (0.55 * out["partial_payload"].clip(0,1) + 0.30 * out["xid_score"].clip(0,1) + 0.15 * out["gpu_metric_drop_ratio"].clip(0,1)).clip(0,1)
    out["mvi_linear"] = (
        weights.missingness * out["missingness_score"] +
        weights.nccl * out["nccl_score"] +
        weights.fabric * out["fabric_score"] +
        weights.detachment * out["detachment_score"] +
        weights.topology * out["topology_criticality"]
    ).clip(0,1)
    out["mvi_score"] = [
        noisy_or([0.72*ms, 0.65*ns, 0.58*fs, 0.80*ds, 0.35*tc, lin])
        for ms, ns, fs, ds, tc, lin in zip(out["missingness_score"], out["nccl_score"], out["fabric_score"], out["detachment_score"], out["topology_criticality"], out["mvi_linear"])
    ]
    def classify(row):
        if row["detachment_score"] >= config.detachment_trigger and row["missingness_score"] >= config.missingness_trigger:
            return "SILENT_DETACHMENT_OR_XID79_PATH"
        if row["nccl_score"] >= config.nccl_trigger:
            return "NCCL_RING_STALL_RISK"
        if row["fabric_score"] >= config.fabric_trigger:
            return "FABRIC_GRAY_FAILURE"
        if row["xid_score"] > 0 or row["ecc_score"] >= 0.5:
            return "ISOLATED_HARDWARE_FAILURE"
        if row["missingness_score"] >= config.missingness_trigger:
            return "TELEMETRY_COLLAPSE"
        if row["mvi_score"] >= config.mvi_trigger:
            return "MVI_WATCH"
        return "OK"
    out["mdd_class"] = out.apply(classify, axis=1)
    def evidence(row):
        ev = []
        if row.get("mata_warning", "ok") != "ok":
            ev.append(str(row.get("mata_warning")))
        if row.get("fabric_warning", "ok") != "ok":
            ev.append(str(row.get("fabric_warning")))
        if row.get("nccl_warning", "ok") != "ok":
            ev.append(str(row.get("nccl_warning")))
        if row["xid_score"] > 0:
            ev.append("XID_PRESENT")
        return "; ".join(ev) if ev else "normal"
    out["evidence"] = out.apply(evidence, axis=1)
    return out

def mvi_summary(mvi: pd.DataFrame) -> pd.DataFrame:
    if mvi.empty:
        return pd.DataFrame()
    return mvi.groupby(["node_id", "gpu_id"]).agg(
        peak_mvi=("mvi_score", "max"),
        peak_missingness=("missingness_score", "max"),
        peak_nccl=("nccl_score", "max"),
        peak_fabric=("fabric_score", "max"),
        peak_detachment=("detachment_score", "max"),
        top_class=("mdd_class", lambda x: x.value_counts().index[0] if len(x) else "OK"),
        alert_frames=("mdd_class", lambda x: int((x != "OK").sum())),
    ).reset_index().sort_values("peak_mvi", ascending=False)

def job_risk_summary(mvi: pd.DataFrame) -> pd.DataFrame:
    if mvi.empty or "job_id" not in mvi.columns:
        return pd.DataFrame()
    return mvi.groupby("job_id").agg(
        peak_mvi=("mvi_score", "max"),
        risky_ranks=("mdd_class", lambda x: int((x != "OK").sum())),
        nodes=("node_id", "nunique"),
        gpus=("component_id", "nunique"),
    ).reset_index().sort_values("peak_mvi", ascending=False)
