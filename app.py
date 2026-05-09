
from __future__ import annotations
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mdd_core import MDDConfig
from mdd_sample_data import SCENARIOS, generate_dataset
from mdd_pipeline import run_mdd_pipeline
from mdd_report import generate_html_report, estimate_gpu_hours_at_risk

st.set_page_config(page_title="FCCT-MDD V5.4", page_icon="🛡️", layout="wide")
st.title("FCCT-MDD V5.4 — Metastability & Detachment Defense")
st.caption("Focused audit product: Missingness, NCCL/RAS stall risk, fabric gray failures, and GPU detachment defense.")

with st.sidebar:
    mode = st.radio("Data source", ["Sample data", "Upload CSVs"], index=0)
    scenario = st.selectbox("Sample scenario", SCENARIOS, index=1, disabled=(mode != "Sample data"))
    steps = st.slider("Sample steps", 80, 300, 160, disabled=(mode != "Sample data"))
    seed = st.number_input("Seed", value=17, step=1, disabled=(mode != "Sample data"))
    st.header("MDD Settings")
    mvi_trigger = st.slider("MVI trigger", 0.05, 0.95, 0.62)
    detachment_trigger = st.slider("Detachment trigger", 0.05, 0.95, 0.72)
    gpu_hour_cost = st.number_input("GPU-hour value/cost USD", value=3.0, step=0.5)
    recovery_minutes = st.number_input("Checkpoint/recovery minutes", value=45.0, step=5.0)

config = MDDConfig(
    mvi_trigger=float(mvi_trigger),
    detachment_trigger=float(detachment_trigger),
    gpu_hour_cost_usd=float(gpu_hour_cost),
    checkpoint_recovery_minutes=float(recovery_minutes),
)

try:
    source = "sample"
    if mode == "Sample data":
        data = generate_dataset(scenario=scenario, steps=int(steps), seed=int(seed))
        source = f"sample:{scenario}"
    else:
        telemetry_file = st.file_uploader("telemetry.csv", type=["csv"])
        scrape_file = st.file_uploader("scrape_health.csv", type=["csv"])
        nccl_file = st.file_uploader("nccl_ras.csv", type=["csv"])
        overlay_file = st.file_uploader("logical_physical_overlay.csv", type=["csv"])
        topology_file = st.file_uploader("topology.csv optional", type=["csv"])
        if not all([telemetry_file, scrape_file, nccl_file, overlay_file]):
            st.info("Upload telemetry, scrape health, NCCL/RAS, and overlay CSVs.")
            st.stop()
        data = {
            "telemetry": pd.read_csv(telemetry_file),
            "scrape": pd.read_csv(scrape_file),
            "nccl": pd.read_csv(nccl_file),
            "overlay": pd.read_csv(overlay_file),
            "topology": pd.read_csv(topology_file) if topology_file else pd.DataFrame(),
        }
        source = "uploaded_csv"
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.exception(e)
    st.stop()

try:
    results = run_mdd_pipeline(data, config)
except Exception as e:
    st.error(f"MDD pipeline failed: {e}")
    st.exception(e)
    st.stop()

mvi = results["mvi"]
roi = estimate_gpu_hours_at_risk(mvi, config)

cols = st.columns(7)
cols[0].metric("Peak MVI", f"{mvi['mvi_score'].max():.3f}" if not mvi.empty else "0")
cols[1].metric("Risky components", roi["risky_components"])
cols[2].metric("Affected GPUs est.", roi.get("affected_components_estimate", 0))
cols[3].metric("GPU-hours at risk", roi["gpu_hours_at_risk"])
cols[4].metric("Value at risk", f"${roi['estimated_usd']}")
top_class = mvi.sort_values("mvi_score", ascending=False).iloc[0]["mdd_class"] if not mvi.empty else "OK"
cols[5].metric("Top class", str(top_class)[:18])
cols[6].metric("Recommendations", len(results["recommendations"]))

tabs = st.tabs(["Executive", "MATA Missingness", "NCCL/RAS", "Fabric", "MVI Timeline", "Overlay", "Recommendations", "Audit Report", "Raw Data"])

with tabs[0]:
    st.subheader("Executive View")
    st.write("V5.4 is not a generic GPU dashboard. It detects telemetry collapse, fabric gray failure, NCCL stall risk, and detachment paths.")
    top = mvi.sort_values("mvi_score", ascending=False).head(20)
    st.dataframe(top[["timestamp", "node_id", "gpu_id", "job_id", "rank", "mvi_score", "mdd_class", "evidence"]], use_container_width=True)
    fig, ax = plt.subplots()
    timeline = mvi.groupby("timestamp")["mvi_score"].max().reset_index()
    ax.plot(timeline["timestamp"], timeline["mvi_score"], label="Peak MVI")
    ax.axhline(config.mvi_trigger, linestyle="--", label="trigger")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("MATA: Missingness-Aware Telemetry Assessment")
    st.write("Missing metrics are treated as signals, not ignored as bad data.")
    st.dataframe(results["mata_summary"], use_container_width=True)
    st.dataframe(results["mata"], use_container_width=True)
    if not results["mata"].empty:
        fig, ax = plt.subplots()
        topnode = results["mata"].sort_values("missingness_score", ascending=False)["node_id"].iloc[0]
        sub = results["mata"][results["mata"]["node_id"] == topnode]
        ax.plot(sub["timestamp"], sub["missingness_score"], label=f"{topnode} missingness")
        ax.plot(sub["timestamp"], sub["gpu_metric_drop_ratio"], label="gpu metric drop")
        ax.tick_params(axis="x", labelrotation=90)
        ax.legend()
        st.pyplot(fig)

with tabs[2]:
    st.subheader("NCCL/RAS Stall Signals")
    st.dataframe(results["nccl_summary"], use_container_width=True)
    st.dataframe(results["nccl"].sort_values("nccl_score", ascending=False).head(200), use_container_width=True)

with tabs[3]:
    st.subheader("Fabric Gray-Failure Signals")
    st.dataframe(results["fabric_summary"], use_container_width=True)
    st.dataframe(results["fabric"].sort_values("fabric_score", ascending=False).head(200), use_container_width=True)

with tabs[4]:
    st.subheader("MVI: Metastability Vulnerability Index")
    st.dataframe(results["mvi_summary"], use_container_width=True)
    fig, ax = plt.subplots()
    for col in ["missingness_score", "nccl_score", "fabric_score", "detachment_score", "mvi_score"]:
        ts = mvi.groupby("timestamp")[col].max().reset_index()
        ax.plot(ts["timestamp"], ts[col], label=col)
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend()
    st.pyplot(fig)
    st.download_button("Download MVI CSV", mvi.to_csv(index=False).encode(), "fcct_mdd_v5_4_mvi.csv", "text/csv")

with tabs[5]:
    st.subheader("Logical-to-Physical Overlay")
    st.write("Maps job/rank to node/GPU/fabric path.")
    st.dataframe(results["overlay_summary"], use_container_width=True)
    st.dataframe(results["overlay"], use_container_width=True)

with tabs[6]:
    st.subheader("Shadow Recommendations")
    st.dataframe(results["recommendations"], use_container_width=True)

with tabs[7]:
    st.subheader("14-Day Audit Report Template")
    html = generate_html_report(
        mvi=results["mvi"],
        mata_summary=results["mata_summary"],
        fabric_summary=results["fabric_summary"],
        nccl_summary=results["nccl_summary"],
        job_summary=results["job_risk_summary"],
        recs=results["recommendations"],
        config=config,
        source=source,
    )
    st.download_button("Download HTML audit report", html.encode(), "fcct_mdd_v5_4_audit_report.html", "text/html")
    st.components.v1.html(html, height=700, scrolling=True)

with tabs[8]:
    st.subheader("Telemetry")
    st.dataframe(data["telemetry"].head(2000), use_container_width=True)
    st.subheader("Scrape health")
    st.dataframe(data["scrape"].head(2000), use_container_width=True)
    st.subheader("NCCL/RAS")
    st.dataframe(data["nccl"].head(2000), use_container_width=True)
    st.subheader("Overlay")
    st.dataframe(data["overlay"], use_container_width=True)
