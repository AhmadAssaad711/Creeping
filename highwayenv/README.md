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
| `results/` | Trained models, metrics CSVs, logs and videos, including tracked historical artifacts. |

## Running

Use the lab's shared environment (see the root README). It installs this
package, so `from creeping.route_attention import ...` works from any folder,
and every runner's `--results-dir` defaults to the matching folder under
`results/` regardless of the working directory. For example:

```bash
python research/creeping/runners/run_route_attention_creeping_experiments.py --help
```

Report notebooks link their videos as `../results/...`, so open them from
`reports/`. The DQN baseline also needs the local `infrastructure/rl-agents`
framework and writes under its `scripts/out/` folder.

## Benchmark conventions

The dense route benchmark is 12 initial vehicles, spawn probability 0.5,
duration 22 seconds and `negotiating_traffic=False`; straight maps to `o2`,
left to `o1`. Learned policies need their matching checkpoint and observation
contract. Use explicit CPU settings and small budgets for smoke checks, and do
not run report builders over frozen reports or overwrite historical results
during a path check.

## Other creeping work

SUMO and CARLA creeping stay in
[`decision-making/negotiation-and-creeping`](../decision-making/negotiation-and-creeping/README.md):
they use other simulators and action contracts. The orchestrator's rule-based
creeping skill is `av-orchestrator/skills/creeping/`.

[Individual scripts](../../docs/catalogs/intersections.md) ·
[Workflow and caveats](../../docs/modules/intersections.md).
