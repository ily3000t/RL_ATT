import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_progress_retry import verify_v0_reference
sys.path.pop(0)


class MechanismPrefixRegressionTests(unittest.TestCase):
    def test_extended_traffic_accepts_exact_prefix_but_rejects_changed_seed_action_or_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root, paths = Path(folder), []
            for label, episodes in (("smoke", 2), ("full", 10)):
                batch = dict(status="passed", git_commit=label, runs=[])
                for checkpoint in range(5):
                    directory = root / (label + str(checkpoint))
                    (directory / "attack").mkdir(parents=True)
                    manifest = dict(status="passed", python_runtime="same", pip_freeze="same", sumo_version="same")
                    entry = dict(run_seed=checkpoint, attack=dict(name="ours_return", parameters=dict(max_attempts=3)),
                                 results_directory="attack", effective_seeds=dict(attack_seed=0, episode_sumo_seeds=list(range(episodes))),
                                 checkpoint_sha256=str(checkpoint), weights_sha256=str(checkpoint))
                    (directory / "manifest.json").write_text(json.dumps(manifest))
                    (directory / "evaluation.json").write_text(json.dumps(dict(runs=[entry])))
                    rows = [dict(episode=e + 1, action=1, attack_cost=dict(wall_seconds=episodes)) for e in range(episodes)]
                    (directory / "attack/steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                    batch["runs"].append(dict(run_dir=str(directory)))
                path = root / (label + ".json")
                path.write_text(json.dumps(batch))
                paths.append(path)
            with patch("analyze_progress_retry.ROOT", root):
                proof = verify_v0_reference(paths[1], paths[0], names=("ours_return",), prefix_episodes=2)
                self.assertEqual(proof["verified_steps"], 10)
                with self.assertRaises(ValueError):
                    verify_v0_reference(paths[1], paths[0], names=("ours_return",))
                entry_path = root / "full0/evaluation.json"
                original = json.loads(entry_path.read_text())
                for kind in ("traffic", "attack_seed", "parameters"):
                    changed = copy.deepcopy(original)
                    entry = changed["runs"][0]
                    if kind == "traffic":
                        entry["effective_seeds"]["episode_sumo_seeds"][0] = 999
                    elif kind == "attack_seed":
                        entry["effective_seeds"]["attack_seed"] = 1
                    else:
                        entry["attack"]["parameters"]["max_attempts"] = 1
                    entry_path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        verify_v0_reference(paths[1], paths[0], names=("ours_return",), prefix_episodes=2)
                entry_path.write_text(json.dumps(original))
                step_path = root / "full0/attack/steps.jsonl"
                rows = [json.loads(line) for line in step_path.read_text().splitlines()]
                rows[0]["action"] = 2
                step_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                with self.assertRaisesRegex(ValueError, "behavior changed"):
                    verify_v0_reference(paths[1], paths[0], names=("ours_return",), prefix_episodes=2)
