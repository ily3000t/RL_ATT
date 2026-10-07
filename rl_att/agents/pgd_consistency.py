"""Independent Clean-matched actor KL regularization baseline; no BO or dual."""

import copy
import torch
import torch.nn.functional as F
from oarl import device

from .clean_victim import CleanVictimAgent
from rl_att.defenses.pgd_consistency import categorical_kl, search, validate_config
from rl_att.training.state import require


class PGDConsistencyAgent(CleanVictimAgent):
    def configure_consistency(self, config, rng, resources):
        self.consistency_config = validate_config(config)
        self.consistency_rng, self.consistency_resources = rng, resources
        self.consistency_counts = dict(updates=0, gradient_evaluations=0, search_actor_forward_calls=0,
                                       search_observation_rows=0, regularizer_actor_forward_calls=0,
                                       regularizer_observation_rows=0, candidate_observations=0)
        self.last_consistency = None

    def training_state_dict(self):
        return dict(config=copy.deepcopy(self.consistency_config), counts=copy.deepcopy(self.consistency_counts),
                    last=copy.deepcopy(self.last_consistency), attack_rng=copy.deepcopy(self.consistency_rng.get_state()))

    def load_training_state_dict(self, state):
        require(state["config"] == self.consistency_config and set(state["counts"]) == set(self.consistency_counts),
                "PGD consistency training schema mismatch")
        require(all(type(value) is int and value >= 0 for value in state["counts"].values()), "Invalid PGD training counts")
        self.consistency_counts, self.last_consistency = copy.deepcopy(state["counts"]), copy.deepcopy(state["last"])
        self.consistency_rng.set_state(state["attack_rng"])

    def train_model(self):
        require(hasattr(self, "consistency_config"), "Configure PGD consistency before training")
        config = self.consistency_config
        if config["coefficient"] == 0 or config["epsilon"] == 0 or (config["multiplicative"] == 0 and config["additive"] == 0):
            # Exact reduction: no search, extra forwards, RNG draws or zero-term backward.
            return super(PGDConsistencyAgent, self).train_model()
        batch = self.replay_buffer.sample(self.batch_size)
        obs1, obs2 = batch['obs1'], batch['obs2']
        acts, rews, done = batch['acts'], batch['rews'], batch['done']
        _, prob = self.select_action_batch(obs1)
        _, prob_next = self.select_action_batch(obs2)
        q1 = self.qf1(obs1).gather(1, acts.long()).squeeze(1)
        q2 = self.qf2(obs1).gather(1, acts.long()).squeeze(1)
        min_q_next = torch.min(self.qf1_target(obs2), self.qf2_target(obs2)).to(device)
        v_backup = (prob_next * min_q_next).sum(dim=-1)
        q_backup = rews + self.gamma * (1 - done) * v_backup
        with torch.no_grad():
            min_q_pi = torch.min(self.qf1(obs1), self.qf2(obs1)).to(device)
        base_loss = (-(prob * min_q_pi).sum(dim=-1)).mean()
        adversarial, measured = search(self.actor, obs1, prob, self.consistency_rng, config)
        perturbed_probability = self.actor(adversarial, softmax_dim=-1)
        regularizer = categorical_kl(prob.detach(), perturbed_probability, config["probability_floor"]).clamp(min=0).mean()
        actor_loss = base_loss + config["coefficient"] * regularizer
        require(bool(torch.isfinite(actor_loss)), "Non-finite consistency actor loss")
        # Raw audit probes are ephemeral; do not add them to inference or resume state.
        self.last_consistency_trace = dict(clean_observation=obs1.detach().numpy().copy(),
            adversarial_observation=adversarial.numpy().copy(), clean_probability=prob.detach().numpy().copy(),
            adversarial_probability=perturbed_probability.detach().numpy().copy())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        qf1_loss, qf2_loss = F.mse_loss(q1, q_backup.detach()), F.mse_loss(q2, q_backup.detach())
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        self.soft_target_update(self.qf1, self.qf1_target)
        self.soft_target_update(self.qf2, self.qf2_target)
        measured.update(base_actor_loss=float(base_loss.detach().item()), consistency_loss=float(regularizer.detach().item()),
                        total_actor_loss=float(actor_loss.detach().item()), coefficient=config["coefficient"], batch_size=self.batch_size)
        for key in ("gradient_evaluations", "search_actor_forward_calls", "search_observation_rows", "candidate_observations"):
            self.consistency_counts[key] += measured[key]
        self.consistency_counts["updates"] += 1
        self.consistency_counts["regularizer_actor_forward_calls"] += 1
        self.consistency_counts["regularizer_observation_rows"] += self.batch_size
        self.consistency_resources.add("candidate_observations", measured["candidate_observations"])
        self.last_consistency = measured
        require(all(bool(torch.isfinite(p).all()) for network in (self.actor, self.qf1, self.qf2, self.qf1_target, self.qf2_target)
                    for p in network.parameters()), "Non-finite trained consistency parameters")
