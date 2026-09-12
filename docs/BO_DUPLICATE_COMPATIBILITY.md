# Duplicate BO suggestions in long training

The first controlled five-seed attempt used training commit
`11e1ccc11ef023e7810946cb905a8372f033f216`, batch
`.local/runs/20260912T083846Z-protocol-a-oarl-long`.
Run seed 0 completed 96 episodes and 7,061 updates, then failed inside episode
97 when BO suggested `[u1, u2] = [0.8, -0.05]` more than once in one update.
`bayesian-optimization==1.2.0` raised `KeyError: Data point ... is not unique`
at `optimizer.register`. This is not a collision termination or a NaN failure.
Seeds 1–4 were deliberately stopped after diagnosing this failure so that the
final cohort could restart together from one source commit. All original logs,
partial episode records, manifests and checkpoints are retained. The aborted
cohort is excluded from final aggregate statistics; it is not a completed trial.

The installed dependency's `TargetSpace.register` rejects exact duplicates.
Its own `TargetSpace.probe` instead returns the existing cached value for an
already known point. The OARL loop uses `register` directly and therefore lacks
that handling. An acquisition optimizer can legitimately return a previously
evaluated boundary point, especially as the actor becomes more concentrated.

The compatibility change checks whether the exact suggested coordinates are
already present before registering. A duplicate retains the existing GP
observation. **Every proposal is still evaluated**, including duplicates, so
the five-evaluation budget, original Torch RNG consumption, fresh differentiable
JS tensors, and maximum-JS selection are retained. There is no jitter, random
replacement, extra acquisition search, bound change, attack gating, or loss
change. Actor and replay are fixed throughout this BO search, and the actor has
no stochastic layers, so repeating the same coordinates gives the same target.

For paths without duplicate suggestions, the original full training update and
post-update random states remain bit-identical to frozen upstream in regression
testing. The previously crashing path is now defined by the dependency's
duplicate-caching convention. This is a documented compatibility extension;
the original code has no successful result to compare after the exception.

Tests force all five suggestions to one point and verify five objective calls,
one GP observation, retained autograd and finite nonzero actor gradients. The
experiment recorder also verifies the number of objective evaluations on every
update and records the cumulative number of duplicate suggestions. These checks
diagnose execution; they do not select actions or disable an attack.
