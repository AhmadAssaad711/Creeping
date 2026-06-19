from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
RESULTS = NOTEBOOKS / "results" / "intersectionRouteAttentionCreeping"
VIDEO_ROOT = NOTEBOOKS / "results" / "route_videos" / "benchmark12_05_before_after_10eps"
NOTEBOOK_PATH = NOTEBOOKS / "intersection_12_05_efficiency_safety_report.ipynb"


def read_first_existing(*relative_paths: str) -> pd.DataFrame:
    for relative_path in relative_paths:
        path = RESULTS / relative_path
        if path.exists():
            return pd.read_csv(path)
    tried = ", ".join(relative_paths)
    raise FileNotFoundError(f"None of these result files exist: {tried}")


def read_optional(relative_path: str) -> pd.DataFrame | None:
    path = RESULTS / relative_path
    if path.exists():
        return pd.read_csv(path)
    return None


def read_existing_frames(*relative_paths: str) -> pd.DataFrame:
    frames = []
    for relative_path in relative_paths:
        path = RESULTS / relative_path
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        tried = ", ".join(relative_paths)
        raise FileNotFoundError(f"None of these result files exist: {tried}")
    return pd.concat(frames, ignore_index=True)


def pct(value) -> str:
    return "" if pd.isna(value) else f"{100.0 * float(value):.1f}%"


def num(value, digits: int = 2) -> str:
    return "" if pd.isna(value) else f"{float(value):.{digits}f}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |")
    return "\n".join(lines)


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def video_gallery(policy: str) -> str:
    chunks: list[str] = []
    for maneuver in ["straight", "left"]:
        folder = VIDEO_ROOT / policy / maneuver
        paths = sorted(folder.glob("*.mp4"))
        chunks.append(f"### {policy} / {maneuver}")
        chunks.append(f"`{folder.relative_to(ROOT).as_posix()}`")
        for index, path in enumerate(paths, start=1):
            rel_nb = path.relative_to(NOTEBOOKS).as_posix()
            rel_repo = path.relative_to(ROOT).as_posix()
            chunks.append(
                f'<p><strong>{html.escape(maneuver)} episode {index:02d}</strong><br>'
                f'<video controls width="520" src="{html.escape(rel_nb)}"></video><br>'
                f'<a href="{html.escape(rel_nb)}">{html.escape(rel_repo)}</a></p>'
            )
    return "\n\n".join(chunks)


def add_efficiency_columns(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    system_delay = df["mean_system_delay_proxy_s"].clip(lower=1e-9)
    ego_stop = df["mean_ego_stopped_time_s"].fillna(0.0)
    unsafe_or_no_crossing = (df["collision_rate"] + df["timeout_rate"]).clip(lower=0.0, upper=1.0)

    df["success_per_system_delay"] = df["success_rate"] / system_delay
    df["safe_crossing_efficiency"] = (
        df["success_rate"] * (1.0 - unsafe_or_no_crossing) / (1.0 + system_delay / 50.0 + ego_stop / 5.0)
    )
    df["conflict_discipline_efficiency"] = df["safe_crossing_efficiency"] * (
        1.0 - df["high_speed_zone_rate"].fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    df["negotiation_efficiency_index"] = (
        df["conflict_discipline_efficiency"]
        * df["creep_speed_rate"].fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    df["traffic_delay_per_vehicle_s"] = df["mean_traffic_delay_proxy_s"] / df["mean_traffic_vehicle_seconds"].clip(
        lower=1e-9
    )
    return df


def shielded_training_attempts_table() -> str:
    sources = [
        ("safety fine-tune v1", "straight_safety12_05_smoke_5k_summary.csv"),
        ("interaction-preserving v2", "straight_safety12_05_interaction_v2_5k_summary.csv"),
        ("shielded interaction v3", "shielded_interaction_v3_5000_combined_summary.csv"),
    ]
    rows: list[list[object]] = []
    for label, filename in sources:
        frame = read_optional(filename)
        if frame is None:
            continue
        frame = add_efficiency_columns(frame)
        for _, row in frame.iterrows():
            rows.append(
                [
                    label,
                    row["maneuver"],
                    int(row["episodes"]),
                    pct(row["success_rate"]),
                    pct(row["collision_rate"]),
                    pct(row["timeout_rate"]),
                    num(row["mean_system_delay_proxy_s"]),
                    num(row["safe_crossing_efficiency"], 4),
                    pct(row["high_speed_zone_rate"]),
                    pct(row["creep_speed_rate"]),
                    num(row["mean_ego_stopped_time_s"]),
                ]
            )
    if not rows:
        return "No shielded fine-tune summaries were found yet."
    return md_table(
        [
            "attempt",
            "maneuver",
            "eval eps",
            "success",
            "collision",
            "timeout",
            "system delay",
            "safe-crossing efficiency",
            "high-speed zone",
            "creep-speed",
            "ego stopped s",
        ],
        rows,
    )


def main() -> None:
    summary = read_existing_frames(
        "benchmark12_05_all_20eps_subprocess_summary.csv",
        "benchmark12_0_5_all_20eps_subprocess_summary.csv",
    )
    summary = summary.drop_duplicates(["maneuver", "policy_family"], keep="last").reset_index(drop=True)
    summary = add_efficiency_columns(summary)
    rendered = pd.read_csv(RESULTS / "benchmark12_05_before_after_rendered_10eps.csv")
    forced_zero_12 = read_first_existing(
        "forced_zero_ego_reset_audit_12_0_5_100eps_summary.csv",
        "forced_zero_ego_uncontrollable_reset_audit_12_0_5_100eps_summary.csv",
    )
    assert (summary["initial_vehicle_count"] == 12).all()
    assert (summary["spawn_probability"] == 0.5).all()
    assert (summary["negotiating_traffic"].astype(str).str.lower() == "false").all()
    assert (forced_zero_12["negotiating_traffic"].astype(str).str.lower() == "false").all()

    summary_rows = []
    for _, row in summary.iterrows():
        summary_rows.append(
            [
                row["maneuver"],
                row["policy_family"],
                int(row["episodes"]),
                pct(row["success_rate"]),
                pct(row["collision_rate"]),
                pct(row["timeout_rate"]),
                num(row["mean_survival_time_s"]),
                num(row["mean_ego_delay_proxy_s"]),
                num(row["mean_traffic_delay_proxy_s"]),
                num(row["mean_system_delay_proxy_s"]),
                num(row["mean_creep_zone_speed"]),
                pct(row["high_speed_zone_rate"]),
                pct(row["creep_speed_rate"]),
                pct(row.get("mean_safety_intervention_rate", 0.0)),
            ]
        )
    summary_md = md_table(
        [
            "maneuver",
            "policy",
            "episodes",
            "success",
            "collision",
            "timeout",
            "survival s",
            "ego delay",
            "traffic delay",
            "system delay",
            "creep-zone speed",
            "high-speed zone",
            "creep-speed",
            "shield intervention",
        ],
        summary_rows,
    )

    throughput_rows = []
    for _, row in summary.iterrows():
        throughput_rows.append(
            [
                row["maneuver"],
                row["policy_family"],
                num(row["success_per_system_delay"], 4),
                pct(row["success_rate"]),
                pct(float(row["collision_rate"]) + float(row["timeout_rate"])),
                num(row["mean_system_delay_proxy_s"]),
                num(row["mean_traffic_vehicle_seconds"]),
                num(row["traffic_delay_per_vehicle_s"], 4),
            ]
        )
    throughput_md = md_table(
        [
            "maneuver",
            "policy",
            "success/system-delay",
            "safe crossing",
            "unsafe or no crossing",
            "system delay",
            "traffic vehicle-s",
            "traffic delay / vehicle-s",
        ],
        throughput_rows,
    )

    negotiation_rows = []
    for _, row in summary.iterrows():
        negotiation_rows.append(
            [
                row["maneuver"],
                row["policy_family"],
                num(row["safe_crossing_efficiency"], 4),
                num(row["conflict_discipline_efficiency"], 4),
                num(row["negotiation_efficiency_index"], 4),
                num(row["mean_ego_stopped_time_s"]),
                pct(row["high_speed_zone_rate"]),
                pct(row["creep_speed_rate"]),
            ]
        )
    negotiation_md = md_table(
        [
            "maneuver",
            "policy",
            "safe-crossing efficiency",
            "conflict-discipline efficiency",
            "negotiation efficiency index",
            "ego stopped s",
            "high-speed zone",
            "creep-speed",
        ],
        negotiation_rows,
    )

    audit_rows = []
    for _, row in forced_zero_12.iterrows():
        audit_rows.append(
            [
                "12/0.5 target",
                row["maneuver"],
                int(row["episodes"]),
                int(row["forced_zero_collisions"]),
                pct(row["forced_zero_collision_rate"]),
                int(row["initial_vehicle_count"]),
                row["spawn_probability"],
                int(row.get("forced_zero_steps", 1)),
            ]
        )
    audit_md = md_table(
        [
            "benchmark",
            "maneuver",
            "tested resets",
            "forced-zero collisions",
            "collision rate",
            "initial vehicles",
            "spawn prob.",
            "stationary steps",
        ],
        audit_rows,
    )

    render_rows = []
    render_summary = rendered.groupby(["policy_family", "maneuver"]).agg(
        episodes=("episode", "count"),
        success=("success", "mean"),
        collision=("collided", "mean"),
        ego_delay=("ego_delay_proxy_s", "mean"),
        traffic_delay=("traffic_delay_proxy_s", "mean"),
        system_delay=("system_delay_proxy_s", "mean"),
    )
    for (policy, maneuver), row in render_summary.iterrows():
        render_rows.append(
            [
                policy,
                maneuver,
                int(row["episodes"]),
                pct(row["success"]),
                pct(row["collision"]),
                num(row["ego_delay"]),
                num(row["traffic_delay"]),
                num(row["system_delay"]),
            ]
        )
    render_md = md_table(
        ["policy", "maneuver", "videos", "success", "collision", "ego delay", "traffic delay", "system delay"],
        render_rows,
    )

    cells = [
        markdown_cell(
            "# 12/0.5 Dense Intersection: Creeping, Efficiency, and Safety Constraint\n\n"
            "This report supersedes lower-density summaries for the target benchmark. The benchmark floor is "
            "`initial_vehicle_count=12`, `spawn_probability=0.5`, `duration=22`, and `negotiating_traffic=False`. "
            "No result below 12/0.5 is used as target evidence here."
        ),
        markdown_cell(
            "## Efficiency Metrics\n\n"
            "Creeping should not be judged only by collision rate. A policy can be safe by freezing, which is not negotiation. "
            "The added efficiency metrics are ego stopped time, ego delay proxy, traffic vehicle-seconds, traffic delay proxy, "
            "system delay proxy, and success per unit system delay."
        ),
        markdown_cell("## 20-Episode Crash-Tolerant Benchmark at 12/0.5\n\n" + summary_md),
        markdown_cell(
            "## Raw Throughput Efficiency\n\n"
            "This view treats every episode as a traffic system. It uses total system delay, traffic vehicle-seconds, "
            "and traffic delay per vehicle-second. It intentionally exposes the trap: a policy that crashes quickly can look "
            "fast on delay metrics.\n\n"
            + throughput_md
        ),
        markdown_cell(
            "## Negotiation-Efficiency Metrics\n\n"
            "To answer whether creeping is efficient for negotiation, I added a second view. "
            "`safe-crossing efficiency` rewards success and penalizes collision/timeout, system delay, and ego stopped time. "
            "`conflict-discipline efficiency` additionally penalizes high-speed movement inside the conflict zone. "
            "`negotiation efficiency index` also requires actual creeping-speed behavior in the conflict zone. "
            "These are diagnostic scores, not physics constants; their job is to stop a crash-fast baseline from winning just because it fails quickly.\n\n"
            + negotiation_md
        ),
        markdown_cell(
            "## Interpretation\n\n"
            "At 12/0.5, learned creeping is not raw-throughput efficient: it creates more total delay because it spends time negotiating. "
            "It is more efficient once negotiation is defined as safe crossing with conflict-zone discipline, especially compared with the "
            "fast baseline whose low delay mostly comes from crashing early. The safety-ellipse shield improves some risk measures but "
            "adds stops and timeouts, so it is not yet the right final controller. The progress-assist shield was worse in this 12/0.5 run "
            "because it mostly produced timeouts. The next required fix is training inside the shielded 12/0.5 dynamics with the "
            "interaction-focused reward."
        ),
        markdown_cell(
            "## Forced-Zero Reset Audit\n\n"
            "This audit forces ego speed to `0 m/s` immediately after reset and then steps the slowest action. "
            "If a collision still occurs, that reset is not preventable by an ego-only policy. "
            "The 12/0.5 seed slice below did not show such unavoidable stationary-ego collisions. "
            "So the remaining failures should be treated as policy/shield failures unless a larger 12/0.5 reset audit finds otherwise.\n\n"
            + audit_md
        ),
        markdown_cell(
            "## Shielded Fine-Tune Attempts\n\n"
            "These runs fine-tune the calibrated creeping policy inside the safety-ellipse wrapper with the interaction-focused reward. "
            "The short v3 run used both maneuvers at 12/0.5 for 5k additional PPO steps. The result is not a solution: straight improves "
            "collision rate relative to the unshielded 12/0.5 learned policy but loses the creeping signature; left-turn training mostly "
            "turns collision risk into timeout/deadlock. This is the current evidence that the final controller needs a deeper shield-aware "
            "training run or a less brittle safety projection, not just a small fine-tune.\n\n"
            + shielded_training_attempts_table()
        ),
        markdown_cell("## Rendered 12/0.5 Video Metrics\n\n" + render_md),
        markdown_cell(f"## Video Root\n\n`{VIDEO_ROOT.relative_to(ROOT).as_posix()}`"),
        markdown_cell("## Before Constraint Videos\n\n" + video_gallery("before_learned_creeping")),
        markdown_cell("## After Safety-Ellipse Constraint Videos\n\n" + video_gallery("after_learned_safety")),
        markdown_cell(
            "## Current 100% Success Status\n\n"
            "Not achieved. At the required 12/0.5 density, the current 20-episode benchmark is far below 100%. "
            "The current reset audit no longer gives us an easy impossibility excuse for this exact seed slice; the technical work is to make "
            "the learned policy cooperate with the safety constraint without freezing or abandoning creeping."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd()\n"
            "REPO = ROOT.parent if ROOT.name == 'notebooks' else ROOT\n"
            "result_dir = REPO / 'notebooks/results/intersectionRouteAttentionCreeping'\n"
            "summary_paths = [\n"
            "    result_dir / 'benchmark12_05_all_20eps_subprocess_summary.csv',\n"
            "    result_dir / 'benchmark12_0_5_all_20eps_subprocess_summary.csv',\n"
            "]\n"
            "summary = pd.concat([pd.read_csv(path) for path in summary_paths if path.exists()], ignore_index=True)\n"
            "summary = summary.drop_duplicates(['maneuver', 'policy_family'], keep='last').reset_index(drop=True)\n"
            "assert (summary['initial_vehicle_count'] == 12).all()\n"
            "assert (summary['spawn_probability'] == 0.5).all()\n"
            "assert (summary['negotiating_traffic'].astype(str).str.lower() == 'false').all()\n"
            "summary"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
