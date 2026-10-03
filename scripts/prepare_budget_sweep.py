"""Freeze a shared model-computation sweep without modifying attack algorithms."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    directory = ROOT / "configs/evaluation"
    groups = []
    upper = []
    for checkpoint in range(5):
        base = json.loads((directory / ("proposed_development_seed%d.json" % checkpoint)).read_text())
        base["attacks"] = base["attacks"][:3]
        name = "zero_one_budgeted_development_attack0_seed%d.json" % checkpoint
        (directory / name).write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        upper.append("configs/evaluation/" + name)
    groups.append(dict(gradient_cap=400, forward_cap=800, attack_seed=0, kind="upper_zero_one_refresh", configs=upper))
    for gradient in (200, 100):
        for attack_seed in (0, 1, 2):
            files = []
            for checkpoint in range(5):
                original = json.loads((directory / ("proposed_development_seed%d.json" % checkpoint)).read_text())
                progress = json.loads((directory / ("proposed_progress_development_seed%d.json" % checkpoint)).read_text())
                config = copy.deepcopy(original)
                config["attacks"] = original["attacks"][:3] + progress["attacks"][3:]
                config["research_seeds"]["attack_seed"] = attack_seed
                for attack in config["attacks"][1:]:
                    attack["parameters"]["resource_limits"].update(gradient_evaluations=gradient, policy_forward_calls=2 * gradient)
                name = "budget_sweep_g%d_attack%d_seed%d.json" % (gradient, attack_seed, checkpoint)
                (directory / name).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                files.append("configs/evaluation/" + name)
            groups.append(dict(gradient_cap=gradient, forward_cap=2 * gradient, attack_seed=attack_seed,
                               kind="paired_budget_sweep", configs=files))
    protocol = dict(kind="shared_compute_budget_sweep", checkpoint_seeds=list(range(5)), attack_seeds=[0, 1, 2],
                    gradient_forward_caps=[[400, 800], [200, 400], [100, 200]], episodes=10, max_steps=200,
                    research_split_id=10, fixed_new_shadow_transitions=200, fixed_shadow_steps=4000,
                    groups=groups, additional_episodes=1650,
                    interpretation="Coupled gradient/forward sensitivity, not independent effects of each resource or held-out validation")
    path = ROOT / "configs/research/shared_compute_budget_sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    prepare()
