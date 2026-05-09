
from __future__ import annotations
import pandas as pd
from mdd_core import log_norm, component_id

def analyze_fabric(telemetry: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "node_id", "gpu_id"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(f"telemetry missing columns: {sorted(missing)}")
    t = telemetry.copy()
    t["node_id"] = t["node_id"].astype(str)
    t["gpu_id"] = t["gpu_id"].astype(str)
    t["component_id"] = [component_id(a, b) for a, b in zip(t["node_id"], t["gpu_id"])]
    for c in ["pcie_replay", "nvlink_crc", "ib_crc", "packet_drops", "xid_errors", "ecc_errors", "throttle_flag"]:
        if c not in t.columns:
            t[c] = 0
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0)
    t = t.sort_values(["component_id", "timestamp"]).copy()
    for c in ["pcie_replay", "nvlink_crc", "ib_crc", "packet_drops", "ecc_errors"]:
        t[f"{c}_delta"] = t.groupby("component_id")[c].diff().fillna(t[c]).clip(lower=0)
    t["pcie_score"] = log_norm(t["pcie_replay_delta"], 20)
    t["nvlink_score"] = log_norm(t["nvlink_crc_delta"], 20)
    t["ib_score"] = log_norm(t["ib_crc_delta"], 20)
    t["drop_score"] = log_norm(t["packet_drops_delta"], 30)
    t["ecc_score"] = log_norm(t["ecc_errors_delta"], 10)
    t["xid_score"] = (t["xid_errors"] > 0).astype(float)
    t["fabric_score"] = (0.25 * t["pcie_score"] + 0.22 * t["nvlink_score"] + 0.22 * t["ib_score"] + 0.16 * t["drop_score"] + 0.10 * t["xid_score"] + 0.05 * t["ecc_score"]).clip(0, 1)
    def warn(row):
        if row["xid_score"] > 0:
            return "XID_OR_DETACHMENT_SIGNAL"
        if row["fabric_score"] >= 0.65:
            return "FABRIC_GRAY_FAILURE"
        if row["pcie_score"] >= 0.5:
            return "PCIE_DEGRADATION"
        if row["nvlink_score"] >= 0.5 or row["ib_score"] >= 0.5:
            return "LINK_ERROR_ACCUMULATION"
        return "ok"
    t["fabric_warning"] = t.apply(warn, axis=1)
    return t[[
        "timestamp", "node_id", "gpu_id", "component_id",
        "pcie_score", "nvlink_score", "ib_score", "drop_score", "ecc_score", "xid_score",
        "fabric_score", "fabric_warning",
        "pcie_replay_delta", "nvlink_crc_delta", "ib_crc_delta", "packet_drops_delta",
    ]]

def fabric_summary(fabric: pd.DataFrame) -> pd.DataFrame:
    if fabric.empty:
        return pd.DataFrame()
    return fabric.groupby("node_id").agg(
        peak_fabric_score=("fabric_score", "max"),
        warning_events=("fabric_warning", lambda x: int((x != "ok").sum())),
        max_pcie_delta=("pcie_replay_delta", "max"),
        max_nvlink_delta=("nvlink_crc_delta", "max"),
        max_ib_delta=("ib_crc_delta", "max"),
    ).reset_index().sort_values("peak_fabric_score", ascending=False)
