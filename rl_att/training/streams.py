"""Explicit auxiliary randomness that never advances upstream training streams."""

import copy
import numpy as np

from rl_att.utils.seeding import TorchPolicyStream


class AuxiliaryStreams:
    ROLES = {"attack": 1, "belief": 2, "initialization": 3, "sampling": 4}

    def __init__(self, run_seed):
        self.seeds = {name: int(np.random.SeedSequence([20261005, run_seed, index]).generate_state(1)[0]) % (2 ** 31 - 1)
                      for name, index in self.ROLES.items()}
        self.numpy = {name: np.random.RandomState(seed) for name, seed in self.seeds.items()}
        self.torch = {name: TorchPolicyStream(seed) for name, seed in self.seeds.items()}

    def state_dict(self):
        return dict(seeds=copy.deepcopy(self.seeds), numpy={n: r.get_state() for n, r in self.numpy.items()},
                    torch={n: r.state.clone() for n, r in self.torch.items()})

    def load_state_dict(self, state):
        if state["seeds"] != self.seeds or set(state["numpy"]) != set(self.ROLES) or set(state["torch"]) != set(self.ROLES):
            raise ValueError("Auxiliary role seeds/streams changed")
        for name in self.ROLES:
            self.numpy[name].set_state(state["numpy"][name])
            self.torch[name].state = state["torch"][name].clone()
