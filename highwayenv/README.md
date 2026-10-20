# HighwayEnv intersection policies

Source and frozen reports live in `notebooks/`, and their results store is
`notebooks/results/`. Run from this directory so runners' `notebooks/results`
defaults and the reports' `REPO / 'notebooks/results/...'` paths resolve here.

`run_route_attention_creeping_experiments.py` calls route-attention helpers to
build environments, train/load PPO and evaluate. The benchmark runner launches
`route_episode_worker.py` per episode; report builders read saved metrics.
`highway_ppo_creeping_utils.py` implements an earlier formulation. The DQN
notebook uses the separate local `infrastructure/rl-agents` framework.

The shared simulator is `simulation/highway-env`, installed as `highway_env`.
Use explicit CPU/small budgets for smoke checks. The dense route benchmark is
12 initial vehicles, spawn probability 0.5, duration 22 seconds, negotiating
traffic off; straight → `o2`, left → `o1`. Learned policies need their matching
checkpoint and observation contract. Do not run report builders over frozen
reports or overwrite historical results during a path check.

[Individual scripts](../../../../docs/catalogs/intersections.md) ·
[Workflow and caveats](../../../../docs/modules/intersections.md).
