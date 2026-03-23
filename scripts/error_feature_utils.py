from __future__ import annotations

import math


RAW_STAT_FIELDS = [
    "mean",
    "std",
    "min",
    "max",
    "abs_max",
    "skewness",
    "excess_kurtosis",
    "p01",
    "p50",
    "p99",
    "near_zero_ratio",
    "pos_ratio",
    "neg_ratio",
]

DERIVED_STAT_FIELDS = [
    "log10_std",
    "upper_tail",
    "lower_tail",
    "standardized_mean",
    "center_decade",
    "span_decades",
    "upper_excess_es0",
    "lower_excess_es0",
    "upper_excess_es1",
    "lower_excess_es1",
    "upper_excess_es2",
    "lower_excess_es2",
]

FORMAT_DEP_FIELDS = [
    "oor_ratio",
    "clip_high_ratio",
    "clip_low_ratio",
    "rel_qerr_mean",
    "zero_after_quant_ratio",
    "unique_ratio",
]

DEFAULT_EPS = 1e-30


def parse_format_name(name: str) -> tuple[int, int]:
    _, n, es = str(name).split("_")
    return int(n), int(es)


def raw_stats_features(stats_map: dict[str, float]) -> list[float]:
    return [float(stats_map.get(name, 0.0)) for name in RAW_STAT_FIELDS]


def derived_stats_map(stats_map: dict[str, float], eps: float = DEFAULT_EPS) -> dict[str, float]:
    mean = float(stats_map.get("mean", 0.0))
    std = float(stats_map.get("std", 0.0))
    p01 = float(stats_map.get("p01", 0.0))
    p50 = float(stats_map.get("p50", 0.0))
    p99 = float(stats_map.get("p99", 0.0))

    safe_std = std + eps
    abs_p01 = abs(p01) + eps
    abs_p50 = abs(p50) + eps
    abs_p99 = abs(p99) + eps

    feats: dict[str, float] = {
        "log10_std": math.log10(safe_std),
        "upper_tail": (p99 - p50) / safe_std,
        "lower_tail": (p50 - p01) / safe_std,
        "standardized_mean": mean / safe_std,
        "center_decade": math.log10(abs_p50),
        "span_decades": math.log10(abs_p99) - math.log10(abs_p01),
    }

    for es in (0, 1, 2):
        log10_useed = (2**es) * math.log10(2.0)
        feats[f"upper_excess_es{es}"] = max(0.0, math.log10(abs_p99) - log10_useed)
        feats[f"lower_excess_es{es}"] = max(0.0, -log10_useed - math.log10(abs_p01))

    return feats


def derived_stats_features(stats_map: dict[str, float], eps: float = DEFAULT_EPS) -> list[float]:
    feats = derived_stats_map(stats_map, eps=eps)
    return [float(feats[name]) for name in DERIVED_STAT_FIELDS]


def prefixed_raw_stat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in RAW_STAT_FIELDS]


def prefixed_derived_stat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in DERIVED_STAT_FIELDS]


def prefixed_quant_feat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in FORMAT_DEP_FIELDS]
