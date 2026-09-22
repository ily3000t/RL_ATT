"""Shared witnesses, complete-rollout scoring and actual-history search.

Nodes are keyed by the entire episode action prefix, never by observation alone.
All policy work is charged before invocation. Oracle work is admitted atomically
in its worker, including any necessary full-history replay.
"""

import hashlib
import json
import numpy as np
import torch
from .search_budget import BudgetExhausted
from .rollout_objectives import rollout_key, optimizer_value


def witness_seed(attack_seed, episode, history, target, attempt):
    value = json.dumps([attack_seed, episode, list(history), target, attempt], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(value.encode("ascii")).digest()[:4], "little")


class BehaviorSearch:
    def __init__(self, attack, victim, context, observation, history, budget, horizon):
        self.attack, self.victim, self.context = attack, victim, context
        self.oracle, self.budget = context.rollout_oracle, budget
        self.observation = np.asarray(observation, dtype=np.float64).copy()
        self.history, self.horizon = tuple(history), horizon
        self.nodes, self.inversions = {}, {}
        self.trace, self.plans, self.attempt_trace = [], [], []
        self.incomplete = []
        self.inner_cache_hits = 0
        self.exhausted = False

    def node(self, history, obs):
        if history not in self.nodes:
            self.budget.charge(policy_forward_calls=2)
            clean = self.victim.action(obs)
            logits = self.victim.logits(obs).detach().numpy()
            ranking = sorted((a for a in range(3) if a != clean), key=lambda a: (-float(logits[a]), a))
            self.nodes[history] = dict(observation=obs.copy(), clean=clean, ranking=ranking,
                                       attempts=[0, 0, 0], best_margin=[None, None, None],
                                       branches={clean: self.witness(obs, clean, clean, None)})
        node = self.nodes[history]
        if not np.array_equal(node["observation"], obs):
            raise ValueError("Same actual history produced a different observation")
        return node

    @staticmethod
    def witness(adv, action, target, seed):
        return dict(adversarial=adv.copy(), action=action, target=target, seed=seed,
                    visits=0, best_key=None)

    def invert(self, node, history, target, attempt):
        key = (history, target, attempt)
        if key in self.inversions:
            self.inner_cache_hits += 1
            return self.inversions[key]
        # Entire primitive is reserved atomically; no partially charged gradient.
        self.budget.charge(gradient_evaluations=self.attack.inner_steps,
                           policy_forward_calls=self.attack.inner_steps + 2)
        seed = witness_seed(self.context.attack_seed, self.context.episode, history, target, attempt)
        obs = node["observation"]
        scales = self.attack.scales(obs)
        offset = np.random.RandomState(seed).uniform(-self.attack.epsilon, self.attack.epsilon, 16)
        for _ in range(self.attack.inner_steps):
            with torch.enable_grad():
                state = torch.tensor(obs + scales * offset, dtype=torch.float32, requires_grad=True)
                logits = self.victim.logits(state, input_grad=True)
                loss = logits[target] - logits[[a for a in range(3) if a != target]].max()
                gradient = torch.autograd.grad(loss, state)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Non-finite input gradient")
            offset = np.clip(offset + self.attack.step_size * gradient.sign().numpy(),
                             -self.attack.epsilon, self.attack.epsilon)
        adv = obs + scales * offset
        actual = self.victim.action(adv)
        final_logits = self.victim.logits(adv).detach().numpy()
        margin = float(final_logits[target] - max(final_logits[a] for a in range(3) if a != target))
        item = self.witness(adv, actual, target, seed)
        self.inversions[key] = item
        node["attempts"][target] += 1
        old = node["best_margin"][target]
        node["best_margin"][target] = margin if old is None else max(old, margin)
        previous = node["branches"].get(actual)
        if previous is None:
            node["branches"][actual] = item
        elif np.max(np.abs(adv - obs)) < np.max(np.abs(previous["adversarial"] - obs)):
            item["visits"], item["best_key"] = previous["visits"], previous["best_key"]
            node["branches"][actual] = item
        self.attempt_trace.append(dict(history=list(history), target=target, actual=actual,
                                       attempt=attempt, seed=seed, margin=margin, succeeded=target == actual))
        return item

    def retry_targets(self, node, history, targets):
        return [a for a in targets if a not in node["branches"] and node["attempts"][a] < self.attack.max_attempts]

    def choose(self, node, history):
        # One attempt per visit. Retry only after every alternative was tried.
        targets = node["ranking"]
        untried = [a for a in targets if node["attempts"][a] == 0]
        retry = self.retry_targets(node, history, targets)
        candidates = untried or retry
        if candidates:
            target = candidates[0]
            self.invert(node, history, target, node["attempts"][target])
        branches = node["branches"]
        return min(branches.values(), key=lambda w: (w["visits"], w["best_key"] or (), w["action"]))

    def rollout(self, mode, targets=None):
        current = self.oracle.begin()
        if not np.array_equal(current["observation"], self.observation):
            raise ValueError("Oracle block root differs from live observation")
        history, plan, rewards, collisions = self.history, [], [], []
        try:
            for offset in range(self.horizon):
                obs = np.asarray(current["observation"], dtype=np.float64)
                node = self.node(history, obs)
                if mode == "fallback":
                    witness = node["branches"][node["clean"]]
                elif mode == "behavior":
                    witness = self.choose(node, history)
                else:
                    witness = self.invert(node, history, int(targets[offset]), 0)
                reply = self.oracle.budgeted_step(witness["action"], self.budget.remaining())
                if not reply["accepted"]:
                    raise BudgetExhausted("Oracle replay allowance exhausted")
                self.budget.charge(**reply["costs"])
                current = reply["transition"]
                plan.append(dict(observation=obs.copy(), adversarial=witness["adversarial"].copy(),
                                 action=witness["action"], target=witness["target"], seed=witness["seed"], history=history))
                history += (witness["action"],)
                rewards.append(float(current["reward"]))
                collisions.append(bool(current["collision"]))
                if current["done"]:
                    break
        except BudgetExhausted:
            self.exhausted = True
            self.incomplete.append(dict(kind=mode, actions=[p["action"] for p in plan], steps=len(plan)))
            raise
        key = rollout_key(rewards, collisions, self.attack.objective)
        for p in plan:
            branch = self.nodes[p["history"]]["branches"][p["action"]]
            branch["visits"] += 1
            branch["best_key"] = key if branch["best_key"] is None else min(key, branch["best_key"])
        self.plans.append(plan)
        self.trace.append(dict(kind=mode, targets=list(targets) if targets is not None else [p["target"] for p in plan],
                               actions=[p["action"] for p in plan], rewards=rewards, collisions=collisions,
                               value=key[-1], score=list(key), terminated=bool(current["done"]),
                               horizon_truncated=not bool(current["done"])))
        return optimizer_value(key)

    def metadata(self):
        candidates = [t for t in self.trace if t["kind"] != "fallback"]
        sequences = {tuple(t["actions"]) for t in candidates}
        prefixes = {tuple(t["actions"][:i]) for t in candidates for i in range(1, len(t["actions"]) + 1)}
        return dict(candidate_trace=self.trace, incomplete_candidates=self.incomplete,
                    budget_exhausted=self.exhausted, inner_attempt_trace=self.attempt_trace,
                    complete_search_candidates=len(candidates), unique_actual_sequences=len(sequences),
                    duplicate_actual_sequences=len(candidates) - len(sequences), unique_actual_prefixes=len(prefixes),
                    unique_target_sequences=len({tuple(t["targets"]) for t in candidates}),
                    nonterminal_candidates=sum(not t["terminated"] for t in candidates),
                    discovered_nodes=len(self.nodes), discovered_branches=sum(len(n["branches"]) for n in self.nodes.values()),
                    failed_target_attempts=sum(not t["succeeded"] for t in self.attempt_trace),
                    retry_attempts=sum(t["attempt"] > 0 for t in self.attempt_trace))
