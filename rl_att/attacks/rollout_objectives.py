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
    # atan can round to +/- pi/2 for large finite inputs. Keep a float64
    # separation between collision classes even after the final addition.
    # Final plan selection uses the original tuple, not this bounded map.
    if len(key) == 1:
        return key[0]
    score = 2.0 / math.pi * math.atan(key[1])
    interior = 1.0 - 2.0 ** -50
    return 2.0 * key[0] + max(-interior, min(interior, score))
