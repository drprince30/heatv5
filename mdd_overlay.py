
from __future__ import annotations
import pandas as pd
from mdd_core import component_id, minmax01

def prepare_overlay(overlay: pd.DataFrame) -> pd.DataFrame:
    o = overlay.copy()
    for c in ["node_id", "gpu_id", "job_id"]:
        if c in o.columns:
            o[c] = o[c].astype(str)
    if "rank" in o.columns:
        o["rank"] = pd.to_numeric(o["rank"], errors="coerce").fillna(-1).astype(int)
    if "component_id" not in o.columns and {"node_id", "gpu_id"}.issubset(o.columns):
        o["component_id"] = [component_id(a, b) for a, b in zip(o["node_id"], o["gpu_id"])]
    if "rank_importance" not in o.columns:
        o["rank_importance"] = 1.0
    o["rank_importance"] = pd.to_numeric(o["rank_importance"], errors="coerce").fillna(1.0)
    o["topology_criticality"] = 0.0
    for job, g in o.groupby("job_id", sort=False):
        idx = g.index
        vals = g["rank_importance"]
        o.loc[idx, "topology_criticality"] = 0.5 if vals.max() == vals.min() else minmax01(vals)
    return o

def job_overlay_summary(overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return pd.DataFrame()
    return overlay.groupby("job_id").agg(
        ranks=("rank", "nunique"),
        nodes=("node_id", "nunique"),
        gpus=("component_id", "nunique"),
        fabric_groups=("fabric_group", "nunique"),
        switches=("switch_id", "nunique"),
    ).reset_index()
