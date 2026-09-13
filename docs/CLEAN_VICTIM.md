# OARL-matched Clean Victim

The clean victim is trained from scratch after completing the OARL Protocol A
milestone. Its implementation is `rl_att/agents/clean_victim.py`, developed on
`feat/clean-victim`. The original `oarl.Agent` remains the robust victim.

## Objective

Write J for the original batch BO maximum JS, lambda = exp(dual), and
V(s') = sum_a pi(a|s') min(Q1_target, Q2_target)(s',a).

| Component | OARL robust victim | Matched clean victim |
| --- | --- | --- |
| Critic target | r + gamma (1-done) [V(s') - lambda J] | r + gamma (1-done) V(s') |
| Actor loss | mean(lambda J - sum_a pi(a|s) min(Q1,Q2)(s,a)) | -mean(sum_a pi(a|s) min(Q1,Q2)(s,a)) |
| BO proposals | Five per training update | None |
| Dual update | Original constraint loss | None |
| Entropy bonus | No explicit SAC entropy term in upstream | None added |

This is an **OARL-derived clean actor-critic**, not an assertion that upstream
implements standard entropy-regularized SAC. It is not an OARL checkpoint
relabeled clean or an OARL run with a zero BO iteration count.

The subclass inherits the original constructor, actor/critics, replay, action
selection, target updates and actor checkpoint serialization. Its train_model
contains the matched update without BO or either robust penalty. Inherited dual
storage/optimizer objects exist solely to keep the initialization path identical;
they are never used or updated by clean training. Episode records therefore
report dual_multiplier as null and both BO/dual training as disabled in the
manifest. The attack seed role is explicitly marked unused.

## Fair controls and verification

Run seeds 0–4, 400 episodes, at most 200 steps, 128-hidden-unit architecture,
learning rates, replay capacity, batch size, update cadence, checkpoint cadence,
traffic and SUMO/CPU runtime are the same as the completed OARL cohort.
Protocol A uses the same per-run training seed lists and phase-separated held-out
lists. The independent interaction policy RNG prevents removing unused BO
categorical samples from shifting that random stream. Different learned policies
can still produce different trajectories, terminations and update counts.

A regression test compares every actor/critic/target tensor and Torch/NumPy RNG
state after one update against OARL with its robust terms set to zero. The test
also fails if the clean learner invokes BO or a dual optimizer step, verifies
that dual storage is unchanged, and checks that learning actually updates weights.
All 16 tests passed when introducing the clean learner, including the OARL
regression suite.

The same saved-weight/probe checks and independent greedy SUMO evaluator are
used for both victims. Each final episode-400 checkpoint is selected in advance;
no checkpoint is chosen based on held-out performance. No Gate, Zero-One, new
attack method or PPO victim is introduced here.

```powershell
python scripts/run_experiment_batch.py --configs configs/experiments/clean_protocol_a_seed0.json configs/experiments/clean_protocol_a_seed1.json configs/experiments/clean_protocol_a_seed2.json configs/experiments/clean_protocol_a_seed3.json configs/experiments/clean_protocol_a_seed4.json --jobs 5 --name protocol-a-clean-long
python scripts/validate_checkpoints.py --run .local/runs/<clean-batch>/run-0
```

Repeat checkpoint validation for run-1 through run-4. Models and raw data remain
in ignored `.local/` storage. Completed results are reported separately after
training and validation finish; unit-test success alone is not reproduction.
