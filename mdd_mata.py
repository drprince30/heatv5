
from __future__ import annotations
import numpy as np
import pandas as pd
from mdd_core import robust_zscore, sigmoid

def analyze_mata(scrape: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "node_id", "up", "scrape_duration_s", "scrape_samples_scraped", "dcgm_gpu_metric_count", "dcgm_metric_families_count"}
    missing = required - set(scrape.columns)
    if missing:
        raise ValueError(f"scrape data missing columns: {sorted(missing)}")
    s = scrape.copy()
    s["node_id"] = s["node_id"].astype(str)
    for c in ["up", "scrape_duration_s", "scrape_samples_scraped", "dcgm_gpu_metric_count", "dcgm_metric_families_count"]:
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0)
    rows = []
    for node, g in s.groupby("node_id", sort=False):
        g = g.sort_values("timestamp").copy()
        n_base = max(5, min(20, int(len(g) * 0.25)))
        base = g.head(n_base)
        base_samples = max(1.0, float(base["scrape_samples_scraped"].median()))
        base_gpu = max(1.0, float(base["dcgm_gpu_metric_count"].median()))
        base_fam = max(1.0, float(base["dcgm_metric_families_count"].median()))
        dur_z = robust_zscore(g["scrape_duration_s"])
        prev_gpu = g["dcgm_gpu_metric_count"].shift(1).replace(0, np.nan)
        gpu_step_drop = (1 - (g["dcgm_gpu_metric_count"] / prev_gpu)).clip(0, 1).fillna(0)
        for idx, row in g.iterrows():
            sample_drop = max(0.0, 1.0 - float(row["scrape_samples_scraped"]) / base_samples)
            gpu_drop = max(0.0, 1.0 - float(row["dcgm_gpu_metric_count"]) / base_gpu)
            fam_drop = max(0.0, 1.0 - float(row["dcgm_metric_families_count"]) / base_fam)
            duration_spike = sigmoid(float(dur_z.loc[idx]) - 2.0)
            target_down = 1.0 if float(row["up"]) < 0.5 else 0.0
            partial_payload = 1.0 if (row["up"] >= 0.5 and gpu_drop > 0.30) else 0.0
            collapse_score = 0.30 * gpu_drop + 0.22 * sample_drop + 0.15 * fam_drop + 0.16 * duration_spike + 0.12 * float(gpu_step_drop.loc[idx]) + 0.05 * target_down
            warnings = []
            if partial_payload:
                warnings.append("PARTIAL_GPU_PAYLOAD_COLLAPSE")
            if target_down:
                warnings.append("TARGET_DOWN")
            if duration_spike > 0.65:
                warnings.append("SCRAPE_DURATION_SPIKE")
            if gpu_step_drop.loc[idx] > 0.35:
                warnings.append("SUDDEN_GPU_METRIC_DROP")
            rows.append({
                "timestamp": row["timestamp"], "node_id": node,
                "scrape_duration_s": float(row["scrape_duration_s"]),
                "samples_drop_ratio": float(sample_drop),
                "gpu_metric_drop_ratio": float(gpu_drop),
                "family_drop_ratio": float(fam_drop),
                "duration_spike_score": float(duration_spike),
                "gpu_step_drop": float(gpu_step_drop.loc[idx]),
                "target_down": float(target_down),
                "partial_payload": float(partial_payload),
                "missingness_score": float(max(0, min(1, collapse_score))),
                "mata_warning": "; ".join(warnings) if warnings else "ok",
            })
    return pd.DataFrame(rows)

def node_mata_summary(mata: pd.DataFrame) -> pd.DataFrame:
    if mata.empty:
        return pd.DataFrame()
    return mata.groupby("node_id").agg(
        peak_missingness=("missingness_score", "max"),
        mean_missingness=("missingness_score", "mean"),
        partial_payload_events=("partial_payload", "sum"),
        target_down_events=("target_down", "sum"),
        max_gpu_drop=("gpu_metric_drop_ratio", "max"),
    ).reset_index().sort_values("peak_missingness", ascending=False)
