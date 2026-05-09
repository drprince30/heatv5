
# FCCT-MDD V5.4 — Metastability & Detachment Defense

V5.4 is the narrowed product wedge.

It is **not** a generic GPU dashboard. It focuses on:

1. missingness-aware telemetry collapse
2. GPU detachment / XID-79-style paths
3. NCCL/RAS stall signals
4. PCIe/NVLink/IB gray failures
5. logical job/rank → physical GPU/fabric overlay
6. MVI: Metastability Vulnerability Index
7. 14-day audit report / preventable GPU-hour estimate

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

## Stress test

```bash
python stress_test_v5_4.py
```

## Upload CSVs

Required:
- telemetry.csv
- scrape_health.csv
- nccl_ras.csv
- logical_physical_overlay.csv

Optional:
- topology.csv

## Best pitch

"We detect structural telemetry collapse and fabric/job-health degradation that may precede GPU detachment, fail-slow behavior, or NCCL collective stalls."
