# Zero-One SUMO adapter audit

This stage implements a **discrete, state-only adaptation** of Zero-One, not a numerical reproduction of its continuous-control MuJoCo experiments. Root OARL training and environment code remain frozen. No Gate, timing attack, new victim, or proposed method is introduced.

## Sources and provenance

- Read-only reference: `../Zero-OneAttack-main`.
- Upstream: https://github.com/andrewmata13/Zero-OneAttack
- Paper: https://stanleybak.com/papers/bak2024iccps.pdf (ICCPS 2024).
- Artifact: https://stanleybak.com/papers/bak2024iccps_repeatability.pdf.
- The downloaded source has no `.git` and no explicit license was found. A remote HEAD query failed on this host; **no verified upstream commit SHA is asserted**. `ZERO_ONE_LOCAL_PROVENANCE.json` fingerprints the local reference, which must not be confused with a Git revision. No reference source is copied or publicly pushed.
- ZOOpt 0.4.2 is a separate MIT-licensed dependency. Its complete wheel, including license, is hash-pinned under ignored `.local/dependencies/zero-one`, loaded by zipimport. The frozen training environment is not modified.

## Original mechanism and portability limits

`opt_attack/optimization_attack.py`: `generate -> next_adv -> optimize -> ZOOpt Objective(cost) -> reset full MuJoCo state -> PGD toward candidate action -> actual policy action -> environment.step -> total_true_reward`. The outer optimizer searches action sequences; the inner optimizer searches observations that induce those actions. Winning observations are executed in consecutive blocks with action confirmation. This is not a policy-only per-state decision attack.

The reference depends on MuJoCo state/attribute restoration, continuous policy outputs, saved victim/filter objects and AdverTorch. Its PGD call does not explicitly declare loss/targeted parameters or a dependency pin. `custom_sim.py` supplies outer budget 10; environment-specific settings include horizon 20 and two inner steps with step size epsilon. Other environments use different horizons. Hopper/Walker candidate retention and terminal reward shaping are special cases and are not portable SUMO definitions.

## Explicit SUMO protocol

- Same five frozen Clean checkpoints, 16D observations, greedy three-action policy, Normal traffic, Protocol A held-out seeds, 20 episodes and 200 live steps as Stage 3.
- Outer ZOOpt/SRACOS searches 20 categorical action targets (0, 1, 2), at most 10 candidate rollouts per block. The horizon is capped by remaining live steps. If the finite target space is smaller than the budget, evaluate that space once.
- Inner PGD: two steps, one uniform random start, normalized step size 1, maximizing `logit(target) - max(logit(other))`. The categorical logit loss is an explicit adaptation, not an undocumented claim about the original continuous loss.
- Projection uses the common Stage 3 box: `abs(delta_i) <= epsilon * (0.2 * abs(observation_i) + 0.05)`, epsilon 1. No physical clipping. Every live state receives a planned perturbation; no risk gate or timing modification.
- Outer objective is the sum of unmodified SUMO rewards over the candidate horizon, ending on original termination. No extra collision penalty or post-terminal reward is added. Record actual horizon and termination; early termination therefore affects the sum as in an episodic objective.
- Execute the selected feasible observation sequence as a block; verify each planned clean observation, actual action, and next transition against real execution. Stable ties retain the first evaluated minimum.
- Derive outer and inner seeds from attack seed, episode and block/state; preserve global Python/NumPy/Torch RNG states around optimization. Environment randomness never shares optimizer draws.

## Simulator oracle and fair interpretation

SUMO's save/load documentation explicitly excludes some lane-change internal state: https://sumo.dlr.de/docs/Simulation/SaveAndLoad.html. Therefore the oracle uses a separate SUMO process and a private source/config copy. It restores the environment's pre-reset attributes and NumPy state, resets with the same seed, and replays the complete executed action prefix. Cached transitions are keyed by the complete episode action prefix, never just the 16D observation. Replayed and live transitions must agree exactly, including collision events; a mismatch aborts evaluation.

Zero-One has **privileged simulator-query access**, whereas Stage 3 attacks use policy queries/gradients only. Shared checkpoints, perturbation budgets and live seeds make outcome comparisons paired; they do not equalize information or compute access. Report candidate rollouts, gradient/forward evaluations, physical shadow steps, replay steps, cache hits, resets and elapsed time separately. No claim of equal query budget is appropriate.

Stage 3 results may be reused only after all new clean trajectories and frozen checkpoint hashes match, with unchanged environment/runtime and evaluation settings. Source commits for both experiments remain explicit.
