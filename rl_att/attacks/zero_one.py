"""State-only Zero-One: categorical action search and targeted gradient inversion."""

from contextlib import contextmanager, redirect_stdout
import io
import itertools
import math
import random
import time
import numpy as np
import torch
from .box import BoxAttack
from rl_att.utils.zero_one_dependency import load_zoopt


@contextmanager
def isolated_rng(seed):
    python, numpy, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)  # also handles ZOOpt's special case of seed zero
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(tensor)


def derived_seed(*words):
    return int(np.random.SeedSequence(words).generate_state(1)[0])


class ZeroOneAttack(BoxAttack):
    def __init__(self, horizon=20, evaluations=10, inner_steps=2, step_size=1.0,
                 objective="targeted_logit_margin", **kwargs):
        super().__init__(**kwargs)
        if any(type(v) is not int or v < 1 for v in (horizon, evaluations, inner_steps)):
            raise ValueError("Positive integer horizon, evaluations and inner_steps required")
        if evaluations < 4 or not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("ZOOpt requires at least four evaluations and positive PGD step size")
        if objective != "targeted_logit_margin" or self.every_n_steps != 1:
            raise ValueError("State-only block attack requires targeted margin and every-step frequency")
        if self.epsilon <= 0:
            raise ValueError("Use NoAttack for a zero budget simulator protocol")
        self.horizon, self.evaluations, self.inner_steps = horizon, evaluations, inner_steps
        self.step_size = step_size
        self.plan, self.plan_episode, self.next_step = [], None, None

    def invert(self, obs, target, victim, seed):
        scales = self.scales(obs)
        offset = np.random.RandomState(seed).uniform(-self.epsilon, self.epsilon, 16)
        for _ in range(self.inner_steps):
            with torch.enable_grad():
                state = torch.tensor(obs + scales * offset, dtype=torch.float32, requires_grad=True)
                logits = victim.logits(state, input_grad=True)
                others = [i for i in range(3) if i != target]
                loss = logits[target] - logits[others].max()
                grad = torch.autograd.grad(loss, state)[0]
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("Non-finite targeted PGD gradient")
            offset = np.clip(offset + self.step_size * grad.sign().numpy(), -self.epsilon, self.epsilon)
        adv = obs + scales * offset
        return adv, victim.action(adv)

    def optimize(self, observation, victim, context):
        oracle = context.rollout_oracle
        horizon = min(self.horizon, context.remaining_steps)
        before = oracle.counts()
        cost = dict(objective_evaluations=0, policy_forward_calls=0, gradient_evaluations=0,
                    inner_cache_hits=0)
        plans, trace, inversion_cache = [], [], {}
        seed = derived_seed(context.attack_seed, 4, context.episode, context.step)

        def objective(solution):
            targets = [int(x) for x in (solution.get_x() if hasattr(solution, "get_x") else solution)]
            current = oracle.begin()
            if not np.array_equal(current["observation"], observation):
                raise ValueError("Oracle block start differs from live observation")
            plan, rewards = [], []
            for offset, target in enumerate(targets):
                obs = np.asarray(current["observation"], dtype=np.float64)
                key = (obs.tobytes(), target, offset)
                if key not in inversion_cache:
                    state_seed = derived_seed(context.attack_seed, 4, context.episode, context.step + offset, target)
                    inversion_cache[key] = self.invert(obs, target, victim, state_seed)
                    cost["gradient_evaluations"] += self.inner_steps
                    cost["policy_forward_calls"] += self.inner_steps + 1
                else:
                    cost["inner_cache_hits"] += 1
                adv, action = inversion_cache[key]
                current = oracle.step(action)
                rewards.append(current["reward"])
                plan.append(dict(observation=obs, adversarial=adv, action=action, target=target))
                if current["done"]:
                    break
            value = float(sum(rewards))
            plans.append(plan)
            trace.append(dict(targets=targets, actions=[p["action"] for p in plan],
                              rewards=rewards, value=value, terminated=current["done"]))
            cost["objective_evaluations"] += 1
            if cost["objective_evaluations"] > self.evaluations:
                raise ValueError("Outer optimizer exceeded its candidate rollout budget")
            return value

        with isolated_rng(seed), redirect_stdout(io.StringIO()):
            if 3 ** horizon <= self.evaluations:
                for targets in itertools.product(range(3), repeat=horizon):
                    objective(targets)
                optimizer = "finite_space_exhaustive"
            else:
                zoopt = load_zoopt()
                dimension = zoopt.Dimension(horizon, [[0, 2]] * horizon, [False] * horizon)
                zoopt.Opt.min(zoopt.Objective(objective, dimension),
                              zoopt.Parameter(budget=self.evaluations, seed=seed))
                optimizer = "zoopt_0.4.2_sracos_categorical"
        if not trace:
            raise ValueError("Outer search produced no feasible rollout")
        best = min(range(len(trace)), key=lambda i: trace[i]["value"])
        self.plan = plans[best]
        after = oracle.counts()
        cost.update({key: after[key] - before[key] for key in after})
        return cost, dict(optimizer=optimizer, outer_seed=seed, candidate_trace=trace,
                          selected_candidate=best, selected_return=trace[best]["value"], horizon=horizon)

    def __call__(self, observation, victim, context):
        if context.rollout_oracle is None or context.remaining_steps is None or context.remaining_steps < 1:
            raise ValueError("Zero-One requires an isolated exact simulator oracle and remaining horizon")
        start = time.perf_counter()
        metadata = dict(name="zero_one", adaptation="categorical_state_only", gate_enabled=False,
                        objective="sum_unmodified_rollout_rewards", inner_objective="targeted_logit_margin",
                        inner_steps=self.inner_steps, step_size=self.step_size, privileged_simulator_access=True)
        if self.plan_episode != context.episode:
            self.plan, self.plan_episode, self.next_step = [], context.episode, context.step
        if context.step != self.next_step:
            raise ValueError("Zero-One block execution must be consecutive")
        if not self.plan:
            cost, details = self.optimize(np.asarray(observation), victim, context)
            metadata.update(details, planned=True)
        else:
            cost = dict(objective_evaluations=0, policy_forward_calls=0, gradient_evaluations=0,
                        shadow_steps=0, replay_steps=0, cache_hits=0, shadow_resets=0,
                        candidate_rollouts=0, live_verified_steps=0, inner_cache_hits=0)
            metadata["planned"] = False
        selected = self.plan.pop(0)
        if not np.array_equal(selected["observation"], observation):
            raise ValueError("Planned observation differs from actual block execution")
        if victim.action(selected["adversarial"]) != selected["action"]:
            raise ValueError("Planned victim action differs from actual block execution")
        cost["policy_forward_calls"] += 1
        cost["wall_seconds"] = time.perf_counter() - start
        metadata.update(target_action=selected["target"], planned_action=selected["action"])
        self.next_step += 1
        return self.result(observation, selected["adversarial"], cost, metadata)
