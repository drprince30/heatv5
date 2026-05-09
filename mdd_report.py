
from __future__ import annotations
from html import escape
import pandas as pd
from mdd_core import MDDConfig

def table(df: pd.DataFrame, n: int = 25) -> str:
    if df is None or df.empty:
        return "<p>No data.</p>"
    return df.head(n).to_html(index=False, escape=True)

def estimate_gpu_hours_at_risk(mvi: pd.DataFrame, config: MDDConfig) -> dict:
    if mvi.empty:
        return {"risky_components": 0, "affected_components_estimate": 0, "gpu_hours_at_risk": 0.0, "estimated_usd": 0.0}
    risky = mvi[mvi["mvi_score"] >= config.mvi_trigger]
    comps = risky["component_id"].nunique() if "component_id" in risky.columns else 0
    if "job_id" in risky.columns and "component_id" in mvi.columns:
        affected_jobs = risky["job_id"].dropna().unique().tolist()
        affected_components = mvi[mvi["job_id"].isin(affected_jobs)]["component_id"].nunique() if affected_jobs else comps
    else:
        affected_components = comps
    gpu_hours = affected_components * (config.checkpoint_recovery_minutes / 60.0)
    usd = gpu_hours * config.gpu_hour_cost_usd
    return {"risky_components": int(comps), "affected_components_estimate": int(affected_components), "gpu_hours_at_risk": round(float(gpu_hours), 2), "estimated_usd": round(float(usd), 2)}

def generate_html_report(mvi, mata_summary, fabric_summary, nccl_summary, job_summary, recs, config, source="unknown") -> str:
    peak = float(mvi["mvi_score"].max()) if not mvi.empty else 0.0
    top_class = str(mvi.sort_values("mvi_score", ascending=False).iloc[0]["mdd_class"]) if not mvi.empty else "OK"
    roi = estimate_gpu_hours_at_risk(mvi, config)
    return f"""
<html><head><title>FCCT-MDD V5.4 Audit Report</title>
<style>body{{font-family:Arial;margin:32px;line-height:1.45}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:6px}}th{{background:#f2f2f2}}.box{{border:1px solid #ddd;border-radius:8px;padding:14px;margin:12px 0}}.warn{{background:#fff3cd}}</style>
</head><body>
<h1>FCCT-MDD V5.4 — Metastability & Detachment Defense Audit</h1>
<div class="box"><b>Source:</b> {escape(source)}<br><b>Peak MVI:</b> {peak:.3f}<br><b>Top class:</b> {escape(top_class)}<br><b>Risky components:</b> {roi["risky_components"]}<br><b>Estimated affected GPU-hours:</b> {roi["gpu_hours_at_risk"]}<br><b>Estimated compute value at risk:</b> ${roi["estimated_usd"]}</div>
<div class="box warn"><b>Scope:</b> Shadow-mode only. It identifies structural telemetry collapse, fabric degradation, and NCCL/RAS stall risk.</div>
<h2>Recommendations</h2>{table(recs, 40)}
<h2>Job Risk Summary</h2>{table(job_summary, 20)}
<h2>MATA Node Summary</h2>{table(mata_summary, 20)}
<h2>Fabric Summary</h2>{table(fabric_summary, 20)}
<h2>NCCL Summary</h2>{table(nccl_summary, 20)}
<h2>Top MVI Frames</h2>{table(mvi.sort_values("mvi_score", ascending=False), 30)}
<h2>Limitations</h2><ul><li>MVI is a risk index, not a validated probability.</li><li>Customer validation is required before automation.</li><li>Missing telemetry may indicate exporter failure, network partition, or actual GPU detachment.</li></ul>
</body></html>"""
