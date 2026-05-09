
from __future__ import annotations
from pathlib import Path
import pandas as pd
from mdd_core import MDDConfig
from mdd_sample_data import SCENARIOS, generate_dataset
from mdd_pipeline import run_mdd_pipeline

def run_suite(out_dir="stress_results_v5_4", steps=120):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for seed in [7, 17, 31]:
            data = generate_dataset(scenario=scenario, steps=steps, seed=seed)
            results = run_mdd_pipeline(data, MDDConfig())
            mvi = results["mvi"]
            top = mvi.sort_values("mvi_score", ascending=False).iloc[0]
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "peak_mvi": round(float(mvi["mvi_score"].max()), 3),
                "top_class": top["mdd_class"],
                "risky_frames": int((mvi["mvi_score"] >= 0.62).sum()),
                "peak_missingness": round(float(mvi["missingness_score"].max()), 3),
                "peak_nccl": round(float(mvi["nccl_score"].max()), 3),
                "peak_fabric": round(float(mvi["fabric_score"].max()), 3),
                "peak_detachment": round(float(mvi["detachment_score"].max()), 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "v5_4_stress_results.csv", index=False)
    summary = df.groupby("scenario").agg(
        cases=("seed", "count"),
        avg_peak_mvi=("peak_mvi", "mean"),
        common_class=("top_class", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
        avg_risky_frames=("risky_frames", "mean"),
        avg_missingness=("peak_missingness", "mean"),
        avg_nccl=("peak_nccl", "mean"),
        avg_fabric=("peak_fabric", "mean"),
        avg_detachment=("peak_detachment", "mean"),
    ).reset_index()
    summary.to_csv(Path(out_dir) / "v5_4_stress_summary.csv", index=False)
    return summary

if __name__ == "__main__":
    print(run_suite().to_string(index=False))
