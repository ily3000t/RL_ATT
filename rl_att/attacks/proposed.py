"""Ours-v0: execute verified observation witnesses selected by behavior search."""

import math
import time
import numpy as np
from .box import BoxAttack
from .search_budget import SearchBudget, BudgetExhausted
from .behavior_search import BehaviorSearch

DEFAULT_LIMITS = dict(gradient_evaluations=400, policy_forward_calls=800,
                      new_shadow_transitions=200, shadow_steps=4000)


class ProposedAttack(BoxAttack):
    search_kind = "behavior"

    def __init__(self, horizon=20, evaluations=10, inner_steps=2, step_size=1., max_attempts=3,
                 objective="return", resource_limits=None, **kwargs):
        super().__init__(**kwargs)
        if any(type(v) is not int or v < 1 for v in (horizon, evaluations, inner_steps, max_attempts)):
            raise ValueError("Positive integer search parameters required")
        if objective not in ("return", "safety") or not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("Invalid objective or PGD step size")
        if self.every_n_steps != 1 or self.epsilon <= 0:
            raise ValueError("Ours-v0 uses a positive every-step perturbation budget without Gate")
        self.horizon, self.evaluations, self.inner_steps = horizon, evaluations, inner_steps
        self.step_size, self.max_attempts, self.objective = step_size, max_attempts, objective
        self.resource_limits = dict(DEFAULT_LIMITS if resource_limits is None else resource_limits)
        SearchBudget(self.resource_limits)
        if self.resource_limits["policy_forward_calls"] < horizon + 2:
            raise ValueError("Reserve at least horizon execution checks and a clean live fallback")
        self.plan, self.episode, self.next_step, self.history = [], None, 0, []

    def search(self, engine):
        for _ in range(self.evaluations):
            engine.rollout("behavior")

    def optimize(self, observation, victim, context):
        horizon = min(self.horizon, context.remaining_steps)
        limits = dict(self.resource_limits)
        for key in ("gradient_evaluations", "new_shadow_transitions"):
            limits[key] = limits[key] * horizon // self.horizon
        self.budget = SearchBudget(limits)
        self.budget.reserve(policy_forward_calls=horizon)
        # An emergency live clean action remains available even when the first
        # complete fallback cannot fit. Its forward is charged in every condition.
        self.budget.charge(policy_forward_calls=1)
        emergency_action = victim.action(observation)
        engine = BehaviorSearch(self, victim, context, observation, self.history, self.budget, horizon)
        before = context.rollout_oracle.counts()
        try:
            engine.rollout("fallback")
            self.search(engine)
        except BudgetExhausted:
            engine.exhausted = True
        metadata = engine.metadata()
        if engine.plans:
            best = min(range(len(engine.trace)), key=lambda i: tuple(engine.trace[i]["score"]))
            self.plan = engine.plans[best]
            metadata.update(selected_candidate=best, selected_return=engine.trace[best]["value"],
                            selected_score=engine.trace[best]["score"], oracle_unplanned_fallback=False)
        else:
            self.plan = [dict(observation=observation.copy(), adversarial=observation.copy(),
                              action=emergency_action, target=emergency_action, seed=None)]
            metadata.update(selected_candidate=None, selected_return=None, selected_score=None,
                            oracle_unplanned_fallback=True)
        self.unplanned = metadata["oracle_unplanned_fallback"]
        after = context.rollout_oracle.counts()
        cost = {k: after[k] - before.get(k, 0) for k in after}
        cost.update(objective_evaluations=len(engine.trace), inner_cache_hits=engine.inner_cache_hits)
        metadata.update(horizon=horizon, block_start=context.step, search_kind=self.search_kind)
        return cost, metadata

    def __call__(self, observation, victim, context):
        if context.rollout_oracle is None or type(context.remaining_steps) is not int or context.remaining_steps < 1:
            raise ValueError("Requires an isolated budget-aware oracle and remaining steps")
        start = time.perf_counter()
        obs = np.asarray(observation, dtype=np.float64)
        self.scales(obs)
        if self.episode != context.episode:
            if context.step != 0:
                raise ValueError("Start each episode at step zero")
            self.plan, self.episode, self.next_step, self.history = [], context.episode, 0, []
        if context.step != self.next_step:
            raise ValueError("Block execution must be consecutive")
        planned = not self.plan
        previous_used = dict.fromkeys(DEFAULT_LIMITS, 0) if planned else dict(self.budget.used)
        if planned:
            cost, details = self.optimize(obs, victim, context)
        else:
            cost, details = dict(objective_evaluations=0, inner_cache_hits=0), {}
        selected = self.plan.pop(0)
        if not np.array_equal(selected["observation"], obs):
            raise ValueError("Live observation differs from planned observation")
        self.budget.consume_reserved(policy_forward_calls=1)
        if victim.action(selected["adversarial"]) != selected["action"]:
            raise ValueError("Live policy output differs from witness action")
        cost.update({k: self.budget.used[k] - previous_used[k] for k in self.budget.used})
        cost["wall_seconds"] = time.perf_counter() - start
        metadata = dict(details, name=("ours_" if self.search_kind == "behavior" else "zero_one_budgeted_") + self.objective,
                        planned=planned, objective=self.objective, gate_enabled=False,
                        privileged_simulator_access=True, budget=self.budget.snapshot(),
                        target_action=selected["target"], planned_action=selected["action"],
                        witness_seed=selected["seed"], eligible=True,
                        oracle_unplanned_fallback=self.unplanned,
                        attempted=selected["seed"] is not None)
        self.next_step += 1
        self.history.append(selected["action"])
        result = self.result(obs, selected["adversarial"], cost, metadata)
        result.attacked = bool(np.any(result.perturbation != 0))
        return result.validate(obs)
