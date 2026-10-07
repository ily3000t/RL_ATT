"""Validated additional victim references; old frozen registry remains intact."""

import hashlib
import json
from pathlib import Path
from rl_att.training.state import require


def defense_registry(runs, root, engineering=False):
    root = Path(root).resolve()
    victims = []
    for run in runs:
        directory = Path(run).resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        audit_path = directory / "training_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        require(manifest["status"] == "passed" and audit["verified"] and audit["engineering_only"] == engineering and
                audit["git_commit"] == manifest["git_commit"] and audit["config_sha256"] == manifest["config_sha256"] and
                hashlib.sha256(audit_path.read_bytes()).hexdigest() == manifest["training_audit_sha256"], "Unverified defense training")
        checkpoint = audit["checkpoints"][-1]
        require(engineering or (audit["episodes"] == 400 and [c["episode"] for c in audit["checkpoints"]] == [100, 200, 300, 400]),
                "Full defense freezing requires all four checkpoints")
        actor_path = Path(checkpoint["checkpoint"]).resolve()
        require(hashlib.sha256(actor_path.read_bytes()).hexdigest() == checkpoint["checkpoint_sha256"], "Defense actor file changed")
        victims.append(dict(victim="pgd_consistency", run_seed=audit["run_seed"], training_episode=checkpoint["episode"],
            checkpoint=actor_path.relative_to(root).as_posix(), checkpoint_sha256=checkpoint["checkpoint_sha256"],
            weights_sha256=checkpoint["weights_sha256"], training_commit=audit["git_commit"],
            training_config=manifest["config_file"], training_manifest=(directory / "manifest.json").relative_to(root).as_posix(),
            training_manifest_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            training_audit=audit_path.relative_to(root).as_posix(), training_audit_sha256=manifest["training_audit_sha256"],
            engineering_only=engineering, benchmark_eligible=not engineering, defense_config=manifest["config"]["defense"]))
    require(len(victims) == len({v["run_seed"] for v in victims}), "Duplicate defense run seed")
    require(engineering or sorted(v["run_seed"] for v in victims) == list(range(5)), "Freeze all five training seeds together")
    require(len({json.dumps(v["defense_config"], sort_keys=True) for v in victims}) == 1 and
            len({v["training_commit"] for v in victims}) == 1, "Mixed training source or defense configuration")
    return dict(schema_version=1, kind="defense_victim_registry", verified=True, engineering_only=engineering,
                benchmark_eligible=not engineering, checkpoint_type="inference_actor_only", path_base="repository_root",
                checkpoint_selection="last_predeclared" if engineering else "episode_400_predeclared",
                victims=sorted(victims, key=lambda v: v["run_seed"]))


def registry_references(value, allow_engineering=False):
    if value.get("kind") != "defense_victim_registry":
        require(all(v["victim"] in ("clean", "oarl") for v in value["victims"]), "Unknown legacy registry victim")
        return value["victims"]
    require(value.get("verified") is True and value["checkpoint_type"] == "inference_actor_only", "Unverified defense registry")
    engineering = value["engineering_only"]
    require(not engineering or allow_engineering, "Engineering checkpoints cannot enter benchmark evaluation")
    require(value["benchmark_eligible"] == (not engineering) and value["victims"], "Invalid defense registry eligibility")
    if not engineering:
        require(value["checkpoint_selection"] == "episode_400_predeclared" and
                sorted(v["run_seed"] for v in value["victims"]) == list(range(5)), "Complete five-seed defense registry required")
    for victim in value["victims"]:
        require(victim["victim"] == "pgd_consistency" and victim["engineering_only"] == engineering and
                victim["benchmark_eligible"] == (not engineering) and (engineering or victim["training_episode"] == 400),
                "Defense victim eligibility mismatch")
    return value["victims"]
