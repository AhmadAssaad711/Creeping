"""LearnedCreepingSkill: observation parity with training, RNG neutrality and request semantics.

Run with the shared environment: ``python -m pytest research/creeping/tests -q``.
"""

from __future__ import annotations

import numpy as np
import pytest

from creeping.route_attention import (
    ATTENTION_OBS_SIZE,
    RouteAttentionObservationWrapper,
    make_route_intersection_env,
    route_config_for,
)
from creeping.skill import LearnedCreepingSkill, route_destination

SEED = 3


def _env(maneuver="straight", attention=False):
    cfg = route_config_for(maneuver)
    return make_route_intersection_env("creeping", cfg, seed=SEED, use_attention_obs=attention), cfg


def _drive_into_creep_zone(env, skill, max_steps=40):
    """IDLE until the ego is inside the trained approach radius."""
    for _ in range(max_steps):
        if skill.in_creep_zone(env):
            return True
        _, _, terminated, truncated, _ = env.step(1)
        if terminated or truncated:
            return False
    return skill.in_creep_zone(env)


@pytest.mark.parametrize("maneuver, destination", [("straight", "o2"), ("left", "o1")])
def test_route_destination_resolves_to_trained_maneuver(maneuver, destination):
    env, _ = _env(maneuver)
    skill = LearnedCreepingSkill()
    assert route_destination(env) == destination
    assert skill.maneuver(env) == maneuver
    assert skill.model_path(maneuver).exists()
    env.close()


def test_observation_matches_training_wrapper():
    """The skill's own frame + features reproduce the training pipeline on the same state.

    Compared on the *current* state rather than against ``step()``'s return value:
    ``IntersectionEnv.step`` observes before it clears departed and spawns new vehicles, so
    the returned frame can be one vehicle stale relative to the road the skill queries.
    """
    env, _ = _env("straight", attention=True)
    assert isinstance(env, RouteAttentionObservationWrapper)
    skill = LearnedCreepingSkill()
    for _ in range(10):
        env.step(1)
        expected = env.observation(env.unwrapped.observation_type.observe())
        actual = skill.observation(env)
        assert actual.shape == (ATTENTION_OBS_SIZE,)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    env.close()


def test_query_does_not_consume_host_randomness():
    env, _ = _env("straight")
    skill = LearnedCreepingSkill()
    assert _drive_into_creep_zone(env, skill)
    before = env.unwrapped.np_random.bit_generator.state
    skill.get_speed_request(env)
    assert env.unwrapped.np_random.bit_generator.state == before
    env.close()


def test_speed_request_is_on_training_grid_and_one_step_from_current():
    env, cfg = _env("left")
    skill = LearnedCreepingSkill()
    assert _drive_into_creep_zone(env, skill)
    request = skill.get_speed_request(env)
    grid = np.asarray(cfg.target_speeds)
    assert request in grid
    current = int(np.argmin(np.abs(grid - env.unwrapped.vehicle.speed)))
    assert abs(int(np.flatnonzero(grid == request)[0]) - current) <= 1
    env.close()


def test_no_request_outside_creep_zone_or_on_untrained_route():
    env, _ = _env("straight")
    skill = LearnedCreepingSkill()
    env.unwrapped.vehicle.position = np.array([0.0, 90.0])  # far up the approach
    assert not skill.in_creep_zone(env)
    assert skill.get_speed_request(env) is None
    env.close()

    right = make_route_intersection_env("creeping", route_config_for("straight", destination="o3"), seed=SEED)
    assert skill.maneuver(right) is None
    assert skill.get_speed_request(right) is None
    right.close()
