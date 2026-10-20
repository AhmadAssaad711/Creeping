"""Canonical locations of the creeping project's files, independent of the working directory."""

from __future__ import annotations

from pathlib import Path

CREEPING_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = CREEPING_ROOT / "results"

ROUTE_ATTENTION_RESULTS = RESULTS_ROOT / "intersectionRouteAttentionCreeping"
PPO_CREEPING_RESULTS = RESULTS_ROOT / "intersectionPPOCreeping"
NATIVE_PPO_RESULTS = RESULTS_ROOT / "nativeIntersectionPPO12_05"
ROUTE_VIDEOS = RESULTS_ROOT / "route_videos"
RENDER_METRICS = RESULTS_ROOT / "render_metrics"
