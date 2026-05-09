
from __future__ import annotations
from typing import Dict
import pandas as pd
from mdd_core import MDDConfig
from mdd_mata import analyze_mata, node_mata_summary
from mdd_fabric import analyze_fabric, fabric_summary
from mdd_nccl import analyze_nccl, job_nccl_summary
from mdd_overlay import prepare_overlay, job_overlay_summary
from mdd_mvi import compute_mvi, mvi_summary, job_risk_summary
from mdd_recommend import build_recommendations

def run_mdd_pipeline(data: Dict[str, pd.DataFrame], config: MDDConfig = MDDConfig()) -> Dict[str, pd.DataFrame]:
    telemetry = data["telemetry"]
    scrape = data["scrape"]
    nccl = data["nccl"]
    overlay = prepare_overlay(data["overlay"])
    mata = analyze_mata(scrape)
    fabric = analyze_fabric(telemetry)
    nccl_score = analyze_nccl(nccl)
    mvi = compute_mvi(telemetry, mata, fabric, nccl_score, overlay, config)
    recs = build_recommendations(mvi, config)
    return {
        "mata": mata,
        "mata_summary": node_mata_summary(mata),
        "fabric": fabric,
        "fabric_summary": fabric_summary(fabric),
        "nccl": nccl_score,
        "nccl_summary": job_nccl_summary(nccl_score),
        "overlay": overlay,
        "overlay_summary": job_overlay_summary(overlay),
        "mvi": mvi,
        "mvi_summary": mvi_summary(mvi),
        "job_risk_summary": job_risk_summary(mvi),
        "recommendations": recs,
    }
