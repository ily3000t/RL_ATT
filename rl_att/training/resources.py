"""Actual training calls/rows and information-access ledger; no inferred FLOPs."""

from contextlib import contextmanager
import copy
import math
import time

from .state import NETWORKS, OPTIMIZERS, require


class TrainingResources:
    def __init__(self):
        self.counts = dict(real_interaction_steps=0, sumo_reset_warmup_steps=0, primary_updates=0,
                           auxiliary_updates=0, attacked_interaction_steps=0,
                           privileged_branch_steps=0, privileged_replay_steps=0, candidate_observations=0,
                           bo_objective_evaluations=0)
        self.phases = {}

    def add(self, name, count=1):
        require(name in self.counts and type(count) is int and count >= 0, "Invalid training resource count")
        self.counts[name] += count

    @contextmanager
    def measure(self, agent, phase):
        record = self.phases.setdefault(phase, dict(network_forward_calls={}, network_observation_rows={}, optimizer_steps={}, wall_seconds=0.))
        hooks, original_steps = [], {}
        original_objective = agent.js_d_loss
        start = time.monotonic()
        try:
            def objective(*args, **kwargs):
                value = original_objective(*args, **kwargs)
                self.add("bo_objective_evaluations")
                return value
            agent.js_d_loss = objective
            for name in NETWORKS:
                def record_forward(module, args, output, name=name):
                    record["network_forward_calls"][name] = record["network_forward_calls"].get(name, 0) + 1
                    rows = 1 if args[0].dim() == 1 else args[0].shape[0]
                    record["network_observation_rows"][name] = record["network_observation_rows"].get(name, 0) + rows
                hooks.append(getattr(agent, name).register_forward_hook(record_forward))
            for name in OPTIMIZERS:
                optimizer = getattr(agent, name)
                original_steps[name] = optimizer.step
                def step(*args, name=name, **kwargs):
                    value = original_steps[name](*args, **kwargs)
                    record["optimizer_steps"][name] = record["optimizer_steps"].get(name, 0) + 1
                    return value
                optimizer.step = step
            yield
        finally:
            for hook in hooks:
                hook.remove()
            for name, original in original_steps.items():
                getattr(agent, name).step = original
            agent.js_d_loss = original_objective
            record["wall_seconds"] += time.monotonic() - start

    def state_dict(self):
        return copy.deepcopy(dict(counts=self.counts, phases=self.phases))

    @contextmanager
    def measure_component(self, name, network, optimizer, phase="auxiliary_update"):
        """Explicitly measure a registered extra network without selecting its loss."""
        require(bool(name), "Name auxiliary component costs")
        record = self.phases.setdefault(phase, dict(network_forward_calls={}, network_observation_rows={}, optimizer_steps={}, wall_seconds=0.))
        start, original = time.monotonic(), optimizer.step

        def forward(module, args, output):
            record["network_forward_calls"][name] = record["network_forward_calls"].get(name, 0) + 1
            rows = 1 if args[0].dim() == 1 else args[0].shape[0]
            record["network_observation_rows"][name] = record["network_observation_rows"].get(name, 0) + rows

        def step(*args, **kwargs):
            value = original(*args, **kwargs)
            record["optimizer_steps"][name] = record["optimizer_steps"].get(name, 0) + 1
            self.add("auxiliary_updates")
            return value
        hook = network.register_forward_hook(forward)
        optimizer.step = step
        try:
            yield
        finally:
            hook.remove()
            optimizer.step = original
            record["wall_seconds"] += time.monotonic() - start

    def load_state_dict(self, state):
        require(set(state["counts"]) == set(self.counts), "Training ledger schema changed")
        require(all(type(v) is int and v >= 0 for v in state["counts"].values()), "Invalid saved resource counts")
        for phase in state["phases"].values():
            require(math.isfinite(phase["wall_seconds"]) and phase["wall_seconds"] >= 0, "Invalid training wall time")
            require(all(type(v) is int and v >= 0 for key in ("network_forward_calls", "network_observation_rows", "optimizer_steps")
                        for v in phase[key].values()), "Invalid saved network/optimizer costs")
        self.counts, self.phases = copy.deepcopy(state["counts"]), copy.deepcopy(state["phases"])
