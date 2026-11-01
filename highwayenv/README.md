# Creeping at intersections (HighwayEnv)

Learned creeping for unsignalised intersections in the shared HighwayEnv
simulator: route-aware attention PPO, behaviour-cloned creeping, safety
shielding, and the benchmarks and reports that compare them with fast/plain
baselines.

| Folder | Contents |
| --- | --- |
| `src/creeping/` | Importable package. `route_attention` holds route configuration, reward/safety/observation wrappers, attention feature extraction, PPO training, model loading, evaluation and rendering. `ppo_creeping` is the earlier creeping formulation. `paths` defines where results live. |
| `runners/` | Command-line experiments: route-attention training (`run_route_attention_creeping_experiments.py`), the 12/0.5 benchmark (`run_benchmark12_05_subprocess.py` launching `route_episode_worker.py` per episode), shielded fine-tuning, the native PPO baseline and the forced-zero reset audit. |
| `notebooks/` | `intersectionPPOCreeping.ipynb`, the earlier creeping formulation notebook. |
| `reports/` | Frozen report notebooks and `build_12_05_efficiency_report.py`, which regenerates the 12/0.5 report. |
| `baselines/` | Plain accelerate-only PPO (`Intersection.ipynb`) and the `rl-agents` DQN notebook. |
| `results/` | Metrics CSVs, run configs and summaries. Trained models (`.zip`), videos, GIFs and logs are not tracked in this repository; they remain in the `agv_Unstructured_Labs` history under `research/creeping/results/`. |

## Running

The simulator and RL framework are submodules, `third_party/HighwayEnv/` and
`third_party/rl-agents/` (the owner's forks, on their `agv-numpy2` branches).
From the repository root, with Python 3.12:

```bash
git submodule update --init third_party/HighwayEnv third_party/rl-agents
py -3.12 -m venv .venv          # Linux/macOS: python3.12 -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

That installs this package, so `from creeping.route_attention import ...`
works from any folder, and every runner's `--results-dir` defaults to the
matching folder under `results/` regardless of the working directory. For
example:

```bash
python highwayenv/runners/run_route_attention_creeping_experiments.py --help
python -m pytest highwayenv/tests -q
```

Report notebooks link their videos as `../results/...`, so open them from
`reports/`. The videos themselves are not tracked here, so those links only
resolve after a re-render or with the videos copied in from
`agv_Unstructured_Labs`. The DQN baseline also needs the `rl-agents`
submodule and writes under its `scripts/out/` folder.

## Benchmark conventions

The dense route benchmark is 12 initial vehicles, spawn probability 0.5,
duration 22 seconds and `negotiating_traffic=False`; straight maps to `o2`,
left to `o1`. Learned policies need their matching checkpoint and observation
contract. Use explicit CPU settings and small budgets for smoke checks, and do
not run report builders over frozen reports or overwrite historical results
during a path check.

## Other creeping work

The SUMO and CARLA creeping work is at the root of this repository
([`Creeping_Sumo/`](../Creeping_Sumo/) and the CARLA stage beside it): it
uses other simulators and action contracts. The orchestrator wrapper around
`creeping.skill` (`cso/wrappers/learned_creeping.py`) and the rule-based
creeping skill (`skills/creeping/`) live in `av-orchestrator/` inside
[`agv_Unstructured_Labs`](https://github.com/AhmadAssaad711/agv_Unstructured_Labs),
as do the lab-wide
[script catalogue](https://github.com/AhmadAssaad711/agv_Unstructured_Labs/blob/master/docs/catalogs/intersections.md)
and
[workflow notes](https://github.com/AhmadAssaad711/agv_Unstructured_Labs/blob/master/docs/modules/intersections.md).

This folder was split out of `agv_Unstructured_Labs` (`research/creeping/`,
earlier `notebooks/`), with its history kept and generated artifacts
removed.
