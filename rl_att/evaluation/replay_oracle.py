"""Exact action-prefix replay cache, independent of TraCI and attack objectives."""

import copy
import numpy as np


def assert_transition(expected, actual):
    if (not np.array_equal(expected["observation"], actual["observation"])
            or any(expected[k] != actual[k] for k in ("reward", "done", "collision"))):
        raise ValueError("Simulator oracle transition diverged from exact action-history replay")


class ReplayOracle:
    def __init__(self, reset, step, reset_step_cost=None):
        self.reset_fn, self.step_fn = reset, step
        self.reset_step_cost = reset_step_cost
        self.last_reset_steps = None
        self.counts = dict(shadow_steps=0, replay_steps=0, cache_hits=0, shadow_resets=0,
                           candidate_rollouts=0, live_verified_steps=0)

    def start_episode(self, initial):
        self.initial = copy.deepcopy(initial)
        self.cache = {(): copy.deepcopy(initial)}
        self.live = self.cursor = ()
        self.physical = None
        self.last_reset_steps = None
        self._reset()

    def _reset(self):
        actual = self.reset_fn()
        self.counts["shadow_resets"] += 1
        if self.reset_step_cost is not None:
            steps = self.reset_step_cost()
            if type(steps) is not int or steps < 0 or (self.last_reset_steps is not None and steps != self.last_reset_steps):
                raise ValueError("Reset warmup cost changed within an exact replay episode")
            self.last_reset_steps = steps
            self.counts["warmup_steps"] = self.counts.get("warmup_steps", 0) + steps
            self.counts["shadow_steps"] += steps
        assert_transition(self.initial, actual)
        self.physical = ()

    def begin(self):
        self.cursor = self.live
        self.counts["candidate_rollouts"] += 1
        return copy.deepcopy(self.cache[self.cursor])

    def step(self, action):
        if type(action) is not int or action not in (0, 1, 2):
            raise ValueError("Expected a discrete policy action")
        if self.cache[self.cursor]["done"]:
            raise ValueError("Cannot query beyond episode termination")
        target = self.cursor + (action,)
        if target in self.cache:
            self.counts["cache_hits"] += 1
        else:
            if self.physical != self.cursor:
                self._reset()
                for previous in self.cursor:
                    actual = self.step_fn(previous)
                    self.physical += (previous,)
                    self.counts["shadow_steps"] += 1
                    self.counts["replay_steps"] += 1
                    assert_transition(self.cache[self.physical], actual)
            actual = self.step_fn(action)
            self.counts["shadow_steps"] += 1
            self.physical = target
            self.cache[target] = copy.deepcopy(actual)
        self.cursor = target
        return copy.deepcopy(self.cache[target])

    def observe(self, action, actual):
        target = self.live + (action,)
        # A live action must have been simulated before the plan is executed.
        if target not in self.cache:
            raise ValueError("Live action was not evaluated by the simulator oracle")
        assert_transition(self.cache[target], actual)
        self.counts["live_verified_steps"] += 1
        self.live = target

    def budgeted_step(self, action, remaining):
        """Reject before reset/replay/step/cache/cursor mutation, inside the worker."""
        if type(action) is not int or action not in (0, 1, 2):
            raise ValueError("Expected a discrete policy action")
        if self.cache[self.cursor]["done"]:
            raise ValueError("Cannot query beyond episode termination")
        target = self.cursor + (action,)
        fresh = int(target not in self.cache)
        physical_steps = fresh * (1 + (len(self.cursor) + (self.last_reset_steps or 0)
                                       if self.physical != self.cursor else 0))
        costs = dict(new_shadow_transitions=fresh, shadow_steps=physical_steps)
        if any(type(remaining.get(k)) is not int or remaining[k] < 0 for k in costs):
            raise ValueError("Missing oracle resource allowance")
        if any(costs[k] > remaining[k] for k in costs):
            return dict(accepted=False, costs=costs)
        return dict(accepted=True, costs=costs, transition=self.step(action))

    def observe_fallback(self, action, actual):
        """Admit a real clean transition when planning could not finish its fallback.

        Cached transitions retain strict verification. New live observations are
        explicitly unverified; any later physical replay must match them exactly.
        """
        if type(action) is not int or action not in (0, 1, 2):
            raise ValueError("Expected a discrete policy action")
        target = self.live + (action,)
        if target in self.cache:
            self.observe(action, actual)
        else:
            self.cache[target] = copy.deepcopy(actual)
            self.counts["live_unverified_fallback_steps"] = self.counts.get("live_unverified_fallback_steps", 0) + 1
            self.live = target
