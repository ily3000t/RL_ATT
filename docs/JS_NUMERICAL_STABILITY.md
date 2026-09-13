# JS numerical stability in long training

The duplicate-compatible cohort at training commit `9d0086c` finished with
seeds 0, 1, 4 completing all 400 episodes, while seeds 2 and 3 failed during
episodes 299 and 377 after 298 and 376 completed episodes respectively. The
recorder detected non-finite trained parameters. Last completed-episode dual
multipliers were about 73.78 and 22.77, and JS values about 0.000144 and 0.000111.
This cohort is retained, including its successful checkpoints, but does not
constitute a complete five-seed result. No missing results are imputed.

The original float32 expression `p * log(p / ((p+q)/2))` has a reproducible
backward instability in the pinned PyTorch 1.3.1 runtime. For p=q=[1e-25, 1],
the forward JS is exactly zero but the first-component gradient is NaN.
Division backward squares its denominator; very small values underflow.
Shared exact zeros can also produce 0/0 in the forward pass.
A full saturated-actor update with biases [0, -60, -70] reproduces non-finite
actor parameters before the change and finite parameters afterwards.

This demonstrates a concrete failure mechanism in the actual training code.
The previous failed runs did not save their last batch or corrupted parameters,
so the precise internal cause of those two historical failures cannot be
confirmed from those artifacts alone. Future failures now retain batch/probability,
parameter and gradient tensors in ignored diagnostic files.

## Minimal arithmetic extension

When every clean and perturbed probability is at least 1e-18, the exact original
float32 code path runs. When any is smaller, only that JS objective evaluation
uses float64 arithmetic and casts the scalar result back to the policy dtype.
The conservative 1e-18 threshold keeps squared denominators out of the float32
underflow region. It is a precision threshold, not an attack-selection rule:
all proposals, gradients and training updates are still evaluated.

In the float64 path, zeros are represented by 1e-150 inside JS only. This is
smaller than every positive float32 probability, while its square stays normal
in float64. Consequently no positive policy probability is clipped, the actor's
output distribution is unchanged, and 0 log 0 is zero after casting back to
float32 with finite gradients through zero softmax outputs. Bounds, BO budget,
objective, loss signs, dual update, reward and checkpoint cadence do not change.
This is not probability smoothing, gradient clipping or a Gate.

Tests cover identical tiny distributions, shared zeros, zero/subnormal softmax
outputs, recovery of a full saturated-actor update, and bit-identical normal-range
updates against frozen upstream. Arithmetic promotions are counted in each run.
Higher precision can change floating-point rounding and subsequent BO choices;
there is no claim of bit equality after promotion. A new complete five-seed
cohort is required rather than pooling versions or selecting successful seeds.
