"""OARL-matched clean actor-critic, without BO or the dual constraint.

This preserves OARL's architecture, initialization, replay and target updates.
It does not add a SAC entropy term. Unused inherited dual state is never trained.
"""

import torch
import torch.nn.functional as F
from oarl import Agent, device


class CleanVictimAgent(Agent):
    def train_model(self):
        batch = self.replay_buffer.sample(self.batch_size)
        obs1, obs2 = batch['obs1'], batch['obs2']
        acts, rews, done = batch['acts'], batch['rews'], batch['done']

        # Preserve the original batch action-sampling calls and their RNG use.
        _, prob = self.select_action_batch(obs1)
        _, prob_next = self.select_action_batch(obs2)
        q1 = self.qf1(obs1).gather(1, acts.long()).squeeze(1)
        q2 = self.qf2(obs1).gather(1, acts.long()).squeeze(1)
        min_q_next = torch.min(self.qf1_target(obs2), self.qf2_target(obs2)).to(device)
        v_backup = (prob_next * min_q_next).sum(dim=-1)
        q_backup = rews + self.gamma * (1 - done) * v_backup

        with torch.no_grad():
            min_q_pi = torch.min(self.qf1(obs1), self.qf2(obs1)).to(device)
        actor_loss = (-(prob * min_q_pi).sum(dim=-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.soft_target_update(self.qf1, self.qf1_target)
        self.soft_target_update(self.qf2, self.qf2_target)
