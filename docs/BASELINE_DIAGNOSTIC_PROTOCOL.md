# Baseline credibility diagnosis (after Stage 4)

This stage diagnoses existing baselines. It introduces no proposed attack, module ablation, training change, Gate or benchmark tuning. The research order is: verify baseline credibility, analyze failures and define a research question, propose a method, validate its effect, then ablate its new modules.

## Questions and fixed evidence

1. The original run seed selects both a trained checkpoint and an evaluation traffic group. The diagonal results alone cannot attribute seed 2 sensitivity to the victim.
2. Examine paired Clean/attacked trajectories up to the first action divergence, observed lane transitions, SUMO collision events and termination. After divergence, comparing states at the same time is descriptive, not a controlled feature intervention.
3. Reload hash-verified frozen policies to confirm recorded clean/adversarial actions and measure logit margins. Check all perturbations, Zero-One target reachability, selected candidates and oracle confirmation. Target action and realized policy action need not coincide.

## Crossed diagnostic control, fixed before execution

Use all 5 frozen Clean checkpoints on all 5 held-out traffic seed groups. Each cell has the existing 20 episodes, 200-step cap, Normal traffic, original greedy policy and unchanged reward/termination. Evaluate **Clean and the existing deterministic FGSM-margin**, with the same Stage 3 box and every-step frequency. Total: 25 cells × 2 methods × 20 episodes = 1000 episodes.

The checkpoint identity remains `run_seed`/`policy_seed`; explicit `traffic_seed` controls SUMO episode seed derivation and Python/NumPy/Torch evaluation initialization. FGSM consumes no attack randomness; its recorded attack seed remains the checkpoint seed but cannot confound this particular diagnostic. No stochastic-attack factorial claim is made. The original configs without `traffic_seed` must remain exactly unchanged. Diagonal Clean and FGSM trajectories must match their Stage 3 records exactly before interpreting the crossed matrix.

Compare checkpoints within each traffic group and traffic groups within each checkpoint. Summarize cell means, collision event rates, return drops, checkpoint averages over all groups and traffic averages over all checkpoints. These observations concern the existing FGSM configuration and five selected frozen policies, not universal victim robustness, training-seed causality, or Zero-One's entire attribution. No independent-episode statistical significance claim is planned.

## Trace interpretation and reporting

Action 0 requests lane index -1, action 1 requests +1, and action 2 sends no new request (it does not cancel previous commands). Lane index is observation feature 13 multiplied by 10. Use only unperturbed live observations; terminal retained observations must not be interpreted as fresh vehicle state. Observed index transitions are not a complete continuous lane-change trace.

SUMO's original reported collisions can include minGap violations. Existing raw records identify ego involvement/presence, not collision partners or detailed impact geometry. Retain these limits and consult the matching SUMO log where available. Longitudinal post-step TTC/DRAC do not cover all lane-change hazards. Observation distances and lane-position safety gaps have different upstream definitions; do not substitute one for the other.

Read-only diagnostic code and configs are committed before execution. Record Git SHA, commands, versions, input/result hashes and effective seeds. Keep raw diagnostics under `.local`; commit a compact summary and findings. Baseline configs/checkpoints and both upstream source trees remain frozen. No new milestone tag is needed merely for this diagnosis.
