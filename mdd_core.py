
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class MDDConfig:
    mvi_trigger: float = 0.62
    detachment_trigger: float = 0.72
    nccl_trigger: float = 0.65
    fabric_trigger: float = 0.62
    missingness_trigger: float = 0.55
    gpu_hour_cost_usd: float = 3.0
    checkpoint_recovery_minutes: float = 45.0

@dataclass(frozen=True)
class MVIWeights:
    missingness: float = 0.30
    nccl: float = 0.25
    fabric: float = 0.20
    detachment: float = 0.15
    topology: float = 0.10

def component_id(node_id: object, gpu_id: object) -> str:
    return f"{node_id}::gpu{gpu_id}"

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)

def noisy_or(values: Iterable[float]) -> float:
    q = 1.0
    for v in values:
        p = max(0.0, min(1.0, float(v)))
        q *= (1.0 - p)
    return 1.0 - q

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def robust_zscore(values: pd.Series) -> pd.Series:
    x = safe_numeric(values)
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        std = x.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return ((x - x.mean()) / std).fillna(0)
    return (0.6745 * (x - med) / mad).fillna(0)

def log_norm(x: pd.Series, scale: float = 10.0) -> pd.Series:
    vals = safe_numeric(x).fillna(0).clip(lower=0)
    return (np.log1p(vals) / np.log1p(scale)).clip(0, 1)

def minmax01(x: pd.Series) -> pd.Series:
    vals = safe_numeric(x).fillna(0)
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    return ((vals - mn) / (mx - mn)).clip(0, 1)
