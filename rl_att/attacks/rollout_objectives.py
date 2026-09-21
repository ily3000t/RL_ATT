"""Ordering of complete rollouts, without modifying environment rewards."""

import math


def rollout_key(rewards, collisions, objective):
    total = 0.0
    for reward in rewards:
        total += float(reward)
    if not math.isfinite(total):
        raise ValueError("Non-finite rollout return")
    if objective == "return":
        return (total,)
    if objective == "safety":
        return (-int(any(collisions)), total)
    raise ValueError("Unknown rollout objective")


def optimizer_value(key):
    # Disjoint intervals preserve collision-first ordering for unbounded finite
    # rewards. Final plan selection always uses the original tuple, not this map.
    return key[0] if len(key) == 1 else 2.0 * key[0] + 2.0 / math.pi * math.atan(key[1])
