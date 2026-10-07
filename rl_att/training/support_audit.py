"""Raw episode and complete numeric state audit, separate from training worker."""

import hashlib
import json
from pathlib import Path

import torch

from .state import require, tree_digest


EPISODE_KEYS = ("episode", "episode_return", "steps", "terminated", "truncated", "sumo_seed",
                "ego_collision_observed", "sumo_collision_vehicle_ids", "training_updates_total", "trajectory_sha256")


def audit_equivalence(reference, continuous, prefix, resumed):
    require(reference["state_digest"] == continuous["state_digest"] == resumed["state_digest"],
            "Full numeric training/RNG state differs from original/continuous/resumed")
    require(continuous["boundary_digest"] == prefix["state_digest"], "Prefix checkpoint differs from continuous boundary")
    require(reference["episodes"] == continuous["episodes"] == prefix["episodes"] + resumed["episodes"],
            "Episode trajectories differ from original/continuous/resumed")
    require(prefix["end_episode"] == resumed["begin_episode"] and continuous["end_episode"] == resumed["end_episode"],
            "Resume boundary skipped/duplicated episodes")
    require(all(continuous["counts"][k] == prefix["counts"][k] + resumed["counts"][k] for k in continuous["counts"]),
            "Logical counts differ between continuous and split training")
    require(reference["counts"]["real_interaction_steps"] == continuous["counts"]["real_interaction_steps"] and
            reference["counts"]["sumo_reset_warmup_steps"] == continuous["counts"]["sumo_reset_warmup_steps"], "SUMO step counts differ")


def audit_run(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    inputs, groups = {}, []

    def read(file):
        data = file.read_bytes()
        inputs[str(file.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    for victim in manifest["protocol"]["victims"]:
        modes = {}
        defense_checks = {}
        for job in manifest["jobs"]:
            if job["victim"] != victim:
                continue
            require(job["returncode"] == 0, "Failed training support worker")
            directory = Path(job["directory"])
            child = json.loads(read(directory / "manifest.json"))
            require(child["status"] == "passed" and child["git_commit"] == manifest["git_commit"] and
                    child["identity"] == job["identity"], "Mixed/failed training source")
            report = json.loads(read(directory / "support_report.json"))
            mode = report["mode"]
            require(mode == job["mode"] and mode not in modes, "Duplicate/wrong support mode")
            require(hashlib.sha256(read(directory / "canonical.pt")).hexdigest() == report["canonical_sha256"], "Numeric state file changed")
            canonical = torch.load(str(directory / "canonical.pt"), map_location="cpu")
            digest = tree_digest(canonical)
            require(digest == report["canonical_state_sha256"], "Numeric state fingerprint mismatch")
            episodes = [{k: row[k] for k in EPISODE_KEYS} for row in
                        (json.loads(line) for line in read(directory / "episodes.jsonl").decode("utf-8").splitlines())]
            begin = report.get("begin_episode", 0)
            require([e["episode"] for e in episodes] == list(range(begin + 1, report["completed_episodes"] + 1)), "Incomplete support episode grid")
            require(sum(e["steps"] for e in episodes) == report["segment_counts"]["real_interaction_steps"], "Raw steps/cost ledger mismatch")
            require(report["completed_episodes"] == canonical["completed_episodes"] and canonical["updates"] > 0,
                    "No trained updates or wrong final episode")
            boundary = None
            if mode != "reference" or victim == "pgd_consistency":
                final = report["final_training_checkpoint"]
                require(hashlib.sha256(read(Path(final["path"]))).hexdigest() == final["sha256"], "Full resume checkpoint changed")
                payload = torch.load(final["path"], map_location="cpu")
                require(payload["kind"] == "episode_boundary_training_state" and payload["identity"] == child["identity"] and
                        payload["loop"]["completed_episodes"] == canonical["completed_episodes"] and
                        payload["loop"]["updates"] == canonical["updates"] and
                        tree_digest(payload["agent"]) == tree_digest(canonical["agent"]) and
                        tree_digest(payload["global_rng"]) == tree_digest(canonical["global_rng"]) and
                        tree_digest(payload["policy_rng"]) == tree_digest(canonical["policy_rng"]), "Incomplete/changed full training state")
                actor = report["actor_artifact"]
                require(hashlib.sha256(read(Path(actor["checkpoint"]))).hexdigest() == actor["checkpoint_sha256"] and
                        actor["engineering_only"] and not actor["benchmark_eligible"], "Engineering artifact misregistered")
                ledger = report["cumulative_resources"]
                require(ledger == payload["resources"] and ledger["counts"]["primary_updates"] == canonical["updates"] and
                        ledger["counts"]["real_interaction_steps"] == canonical["interactions"] and
                        ledger["counts"]["bo_objective_evaluations"] == (5 * canonical["updates"] if victim == "oarl" else 0),
                        "BO/update/replay cost ledger mismatch")
                steps = ledger["phases"]["primary_update"]["optimizer_steps"]
                require(steps == {n: canonical["updates"] for n in
                        (["actor_optimizer", "qf1_optimizer", "qf2_optimizer"] + (["dual_cst_optimizer"] if victim == "oarl" else []))},
                        "Optimizer step accounting drift")
                if victim == "pgd_consistency":
                    from .consistency_audit import audit_training
                    require(child["comparison_reference"] == report["comparison_reference"] == "independent_repeat",
                            "PGD baseline incorrectly labeled as original-paper reference")
                    rows = [json.loads(line) for line in read(directory / "defense_updates.jsonl").decode("utf-8").splitlines()]
                    defense_checks[mode] = audit_training(rows, child["config"]["defense"], payload["agent"]["defense_training"],
                                                         ledger, report["segment_counts"]["primary_updates"])
                    require(tree_digest(payload["agent"]["defense_training"]["attack_rng"]) ==
                            tree_digest(payload["auxiliary_rng"]["numpy"]["attack"]), "Defense attack RNG differs from auxiliary stream")
                    require(actor["defense_config"] == child["config"]["defense"] and
                            actor["defense_training_counts"] == payload["agent"]["defense_training"]["counts"] and
                            tree_digest(torch.load(actor["checkpoint"], map_location="cpu").state_dict()) ==
                            tree_digest(payload["agent"]["networks"]["actor"]["weights"]),
                            "Exported actor differs from trained model")
                checkpoint = report["boundary_checkpoint"]
                if checkpoint:
                    require(hashlib.sha256(read(Path(checkpoint["path"]))).hexdigest() == checkpoint["sha256"], "Boundary checkpoint changed")
                    saved = torch.load(checkpoint["path"], map_location="cpu")
                    check = dict(agent=saved["agent"], policy_rng=saved["policy_rng"], global_rng=saved["global_rng"],
                                 interactions=saved["loop"]["interactions"], updates=saved["loop"]["updates"],
                                 completed_episodes=saved["loop"]["completed_episodes"])
                    require(tree_digest(check) == checkpoint["canonical_state_sha256"], "Boundary numeric state differs from report")
                    boundary = checkpoint["canonical_state_sha256"]
                if mode == "resumed":
                    require(hashlib.sha256(read(Path(child["resume_path"]))).hexdigest() == child["resume_sha256"], "Resume input changed")
            modes[mode] = dict(state_digest=digest, episodes=episodes, counts=report["segment_counts"],
                               begin_episode=begin, end_episode=report["completed_episodes"], boundary_digest=boundary)
        require(set(modes) == {"reference", "continuous", "prefix", "resumed"}, "Missing support comparison mode")
        audit_equivalence(*(modes[name] for name in ("reference", "continuous", "prefix", "resumed")))
        groups.append(dict(victim=victim, verified=True, episodes=modes["continuous"]["end_episode"],
                           numeric_state_sha256=modes["continuous"]["state_digest"],
                           boundary_state_sha256=modes["prefix"]["state_digest"],
                           split_after_episode=modes["prefix"]["end_episode"], counts=modes["continuous"]["counts"],
                           physical_episodes=sum(len(m["episodes"]) for m in modes.values()),
                           actual_batch_sumo_steps=sum(m["counts"]["real_interaction_steps"] + m["counts"]["sumo_reset_warmup_steps"] for m in modes.values())))
        if defense_checks:
            groups[-1]["defense_checks"] = defense_checks
            groups[-1]["reference_kind"] = "independent_repeat"
    return dict(verified=True, kind="defense_training_support_compatibility", git_commit=manifest["git_commit"],
                source_sha256=inputs, groups=groups, physical_episodes=sum(g["physical_episodes"] for g in groups),
                interpretation=("Short exact-engineering verification on training seeds; PGD reference is an independent repeat, not paper reproduction. No efficacy or final traffic validation"
                                if manifest["protocol"]["kind"] == "pgd_consistency_engineering" else
                                "Short exact-engineering verification on training seeds; not new defense efficacy, full reproduction or final traffic validation"))
