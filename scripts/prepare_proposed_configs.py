"""Freeze Ours-v0 split seeds and smoke/development configs before evaluation.

Run with the legacy experiment Python (NumPy 1.21.6). This creates configurations
only; it never runs or inspects validation or final-test outcomes.
"""

import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def derive(words):
    return int(np.random.SeedSequence(words).generate_state(1)[0]) % (2 ** 31 - 1)


def protocol():
    splits = {str(split): [derive([20260921, 1, split, episode]) for episode in range(count)]
              for split, count in ((10, 10), (20, 20), (30, 50))}
    old = {derive([seed, phase, episode]) for seed in range(5)
           for phase, count in ((0, 400), (1, 20)) for episode in range(count)}
    old.update(range(5))
    old.update(2 * (episode // 2) for episode in range(400))
    new = [s for seeds in splits.values() for s in seeds]
    if len(new) != len(set(new)) or set(new) & old:
        raise ValueError("Seed collision: stop before launching any experiment")
    payload = dict(root_seed=20260921, phase_id=1, episode_index="zero_based", splits=splits,
                   split_names={"10": "development", "20": "validation", "30": "final_test"},
                   attack_seeds=[0, 1, 2], old_unique_seeds_checked=len(old), intersections=[],
                   old_protocols_checked=["controlled_train_5x400", "controlled_evaluation_5x20", "constant_0_to_4", "upstream_paired_400"])
    payload["payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def config(seed, episodes, schedule):
    envelope = dict(norm="observation_scaled_linf", epsilon=1., relative_scale=.2, absolute_scale=.05)
    attacks = [dict(name="none", parameters={}, budget=dict(norm="none"))]
    for name in ("zero_one_budgeted_return", "zero_one_budgeted_safety", "ours_return", "ours_safety"):
        parameters = dict(epsilon=1., relative_scale=.2, absolute_scale=.05, every_n_steps=1,
                          horizon=20, evaluations=10, inner_steps=2, step_size=1., max_attempts=3,
                          objective=name.split("_")[-1], resource_limits=dict(gradient_evaluations=400,
                          policy_forward_calls=800, new_shadow_transitions=200, shadow_steps=4000))
        attacks.append(dict(name=name, parameters=parameters, budget=envelope))
    return dict(victims=["clean"], run_seeds=[seed], episodes=episodes, max_steps=200,
                action_selection="greedy_argmax", gate_enabled=False, sumo_schedule="research_split_v1",
                lookahead_m=1000., metric_percentiles=dict(ttc_percentile=5, drac_percentile=95),
                verify_legacy_no_attack=False, attacks=attacks,
                research_seeds=dict(root_seed=20260921, split_id=10, attack_seed=0,
                                    episode_sumo_seeds=schedule[:episodes]))


def save(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Refusing to overwrite a different frozen configuration: " + str(path))
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    record = protocol()
    save(ROOT / "configs/research_seed_splits.json", record)
    for seed in range(5):
        for label, count in (("smoke", 2), ("development", 10)):
            save(ROOT / ("configs/evaluation/proposed_%s_seed%d.json" % (label, seed)),
                 config(seed, count, record["splits"]["10"]))
    print("Frozen 80 disjoint traffic seeds and 10 development configurations; no experiments executed.")
