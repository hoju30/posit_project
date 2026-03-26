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
]

CURRENT_ES_DERIVED_FIELDS = [
    "upper_excess_current_es",
    "lower_excess_current_es",
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

    return feats


def derived_stats_features(stats_map: dict[str, float], eps: float = DEFAULT_EPS) -> list[float]:
    feats = derived_stats_map(stats_map, eps=eps)
    return [float(feats[name]) for name in DERIVED_STAT_FIELDS]


def current_es_excess_map(
    stats_map: dict[str, float],
    es: int,
    eps: float = DEFAULT_EPS,
) -> dict[str, float]:
    p01 = float(stats_map.get("p01", 0.0))
    p99 = float(stats_map.get("p99", 0.0))
    abs_p01 = abs(p01) + eps
    abs_p99 = abs(p99) + eps
    log10_useed = (2**int(es)) * math.log10(2.0)
    return {
        "upper_excess_current_es": max(0.0, math.log10(abs_p99) - log10_useed),
        "lower_excess_current_es": max(0.0, -log10_useed - math.log10(abs_p01)),
    }


def current_es_excess_features(
    stats_map: dict[str, float],
    es: int,
    eps: float = DEFAULT_EPS,
) -> list[float]:
    feats = current_es_excess_map(stats_map, es=es, eps=eps)
    return [float(feats[name]) for name in CURRENT_ES_DERIVED_FIELDS]


def prefixed_raw_stat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in RAW_STAT_FIELDS]


def prefixed_derived_stat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in DERIVED_STAT_FIELDS]


def prefixed_current_es_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in CURRENT_ES_DERIVED_FIELDS]


def prefixed_quant_feat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in FORMAT_DEP_FIELDS]
