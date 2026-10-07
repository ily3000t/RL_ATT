"""Audit completed PGD runs and describe training; never execute or select models."""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import numpy as np
from run_baseline import git, write_json
from rl_att.training.production_audit import sha, jsonl, audit_episodes
from rl_att.training.protocol import validate_config
from rl_att.training.session import training_seeds
from rl_att.training.state import require
from rl_att.utils.checkpoints import verify_checkpoint
from rl_att.utils.victim_registry import defense_registry, registry_references


def episode_windows(rows, width=100):
    require(rows and len(rows) % width == 0 and [r["episode"] for r in rows] == list(range(1, len(rows) + 1)),
            "Incomplete or reordered episode windows")
    return [dict(first_episode=start + 1, last_episode=start + width, episodes=width,
                 return_mean=float(np.mean([r["episode_return"] for r in rows[start:start + width]])),
                 steps_mean=float(np.mean([r["steps"] for r in rows[start:start + width]])),
                 collisions=sum(r["ego_collision_observed"] for r in rows[start:start + width]),
                 terminated=sum(r["terminated"] for r in rows[start:start + width]))
            for start in range(0, len(rows), width)]


def probe_profile(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 3 and len(p) and np.isfinite(p).all() and
            np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1, atol=1e-6, rtol=0), "Invalid probe probabilities")
    positive = p > 0
    entropy = np.zeros_like(p)
    entropy[positive] = -p[positive] * np.log(p[positive])
    return dict(observations=len(p), greedy_action_counts=np.bincount(p.argmax(1), minlength=3).tolist(),
                entropy_nats_mean=float(entropy.sum(1).mean()), max_probability_mean=float(p.max(1).mean()))


def paired_training(groups):
    """Match run and episode traffic; report descriptive training outcomes only."""
    require(set(groups) == {"clean", "oarl", "pgd_consistency"} and
            all(set(group) == set(range(5)) for group in groups.values()), "Require all five seeds in each training cohort")
    per_seed, cohorts = [], {}
    for seed in range(5):
        expected = [(r["episode"], r["sumo_seed"]) for r in groups["pgd_consistency"][seed]]
        require(len(expected) == 400 and [e for e, s in expected] == list(range(1, 401)), "Incomplete training grid")
        require(all([(r["episode"], r["sumo_seed"]) for r in group[seed]] == expected for group in groups.values()),
                "Training traffic not paired by run and episode")
        values = {name: episode_windows(group[seed])[-1] for name, group in groups.items()}
        per_seed.append(dict(run_seed=seed, last100=values,
                             pgd_minus_clean_return=values["pgd_consistency"]["return_mean"] - values["clean"]["return_mean"],
                             pgd_minus_oarl_return=values["pgd_consistency"]["return_mean"] - values["oarl"]["return_mean"]))
    for name in groups:
        means = [row["last100"][name]["return_mean"] for row in per_seed]
        collisions = sum(row["last100"][name]["collisions"] for row in per_seed)
        cohorts[name] = dict(last100_return_mean=float(np.mean(means)), across_run_sample_sd=float(np.std(means, ddof=1)),
                             collisions=collisions, episodes=500, collision_episode_fraction=collisions / 500.)
    return dict(cohorts=cohorts, per_seed=per_seed, statistics="Descriptive fixed-run training means and sample SD; no test or confidence interval")


class TransitionProfile:
    def __init__(self):
        self.probes, self.bins = [], [dict(action_counts=[0, 0, 0], steps=0, speed_sum=0.) for _ in range(4)]

    def stream(self, source):
        for row in source:
            require(type(row["action"]) is int and row["action"] in (0, 1, 2) and 1 <= row["episode"] <= 400,
                    "Invalid transition action or episode")
            observation = np.asarray(row["observation"], dtype=np.float32)
            if len(self.probes) < 64:
                self.probes.append(observation.copy())
            window_stats = self.bins[(row["episode"] - 1) // 100]
            window_stats["action_counts"][row["action"]] += 1
            window_stats["steps"] += 1
            window_stats["speed_sum"] += float(row["observation"][0]) * 35.
            yield row

    def results(self):
        return [dict(first_episode=100*i+1, last_episode=100*(i+1), steps=b["steps"], action_counts=b["action_counts"],
                     no_new_lane_request_action_fraction=b["action_counts"][2] / b["steps"],
                     pre_action_observation_decoded_speed_mps=b["speed_sum"] / b["steps"])
                for i, b in enumerate(self.bins)]


def analyze(batch_directory):
    require(not git("status", "--porcelain"), "Commit analysis code before auditing")
    initial_head = git("rev-parse", "HEAD")
    sources = {}

    def record(path):
        path = Path(path).absolute()
        require(ROOT in path.parents, "Analysis input outside repository")
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding="utf-8"))

    batch_directory = Path(batch_directory).absolute()
    batch = read(batch_directory / "batch.json")
    require(batch["status"] == "passed" and len(batch["runs"]) == 5 and all(r["returncode"] == 0 for r in batch["runs"]),
            "Incomplete or failed long batch")
    require(subprocess.call(["git", "merge-base", "--is-ancestor", batch["git_commit"], initial_head], cwd=str(ROOT)) == 0,
            "Training execution is not an ancestor of analysis")
    registry_path = Path(batch["frozen_registry"]["path"])
    registry = read(registry_path)
    require(sha(registry_path) == batch["frozen_registry"]["sha256"], "Batch registry changed")
    registry_references(registry)
    regenerated = defense_registry([r["run_dir"] for r in batch["runs"]], ROOT)
    require(registry == regenerated, "Frozen registry no longer matches audited training")
    groups, results = {name: {} for name in ("clean", "oarl", "pgd_consistency")}, []
    for job in batch["runs"]:
        directory = Path(job["run_dir"])
        manifest = read(directory / "manifest.json")
        audit = read(directory / "training_audit.json")
        report = read(directory / "training_report.json")
        seed, config = manifest["config"]["run_seed"], validate_config(manifest["config"])
        config_path = ROOT / manifest["config_file"]
        require(read(config_path) == config and sha(config_path) == manifest["config_sha256"] == batch["configs_sha256"][manifest["config_file"]],
                "Recorded configuration changed")
        committed = json.loads(subprocess.check_output(["git", "show", batch["git_commit"] + ":" + manifest["config_file"]], cwd=str(ROOT)))
        require(config == committed and config["training_stage"] == "full", "Not the preregistered long protocol")
        require(seed not in groups["pgd_consistency"] and manifest["status"] == "passed" and audit["verified"] and
                audit["benchmark_eligible"] and not audit["engineering_only"] and manifest["git_commit"] == audit["git_commit"] == batch["git_commit"] and
                sha(directory / "training_audit.json") == manifest["training_audit_sha256"], "Invalid training eligibility or source")
        # Recheck every file covered by the completed raw loss/checkpoint audit.
        # This does not rerun the already verified multi-gigabyte KL calculation.
        for name, expected in audit["source_sha256"].items():
            path = record(name)
            require(sources[path.relative_to(ROOT).as_posix()] == expected, "Audited source file changed: " + name)
        episodes = [json.loads(line) for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        profile = TransitionProfile()
        counts = audit_episodes(episodes, profile.stream(jsonl(directory / "transitions.jsonl.gz")), config, training_seeds(config))
        require(counts == audit["counts"] and report["cumulative_resources"] == audit["resources"], "Raw training counts changed")
        require([c["episode"] for c in report["checkpoints"]] == [100, 200, 300, 400], "Checkpoint selection changed")
        checkpoints = []
        for checkpoint in report["checkpoints"]:
            actor = checkpoint["actor"]
            verified = verify_checkpoint(actor["checkpoint"], np.stack(profile.probes), actor["probabilities"], actor["weights_sha256"])
            require(verified["checkpoint_sha256"] == actor["checkpoint_sha256"], "Checkpoint file changed")
            checkpoints.append(dict(episode=checkpoint["episode"], checkpoint=str(Path(actor["checkpoint"]).relative_to(ROOT)),
                                    checkpoint_sha256=verified["checkpoint_sha256"], weights_sha256=verified["weights_sha256"],
                                    probe_profile=probe_profile(verified["probabilities"])))
        groups["pgd_consistency"][seed] = episodes
        results.append(dict(run_seed=seed, windows=episode_windows(episodes), transition_windows=profile.results(),
            counts=counts, checkpoints=checkpoints, resources=audit["resources"], defense_counts=report["defense_training"],
            training_wall_seconds=report["training_wall_seconds"], raw_kl_min=audit["raw_kl_min"], raw_kl_max=audit["raw_kl_max"],
            effective_seeds=manifest["effective_seeds"], auxiliary_role_seeds=manifest["auxiliary_role_seeds"],
            runtime=json.loads(manifest["python_runtime"]["stdout"]), sumo_version=manifest["sumo_version"]["stdout"].splitlines()[0],
            training_command=manifest["wrapper_command"], source_snapshots_unchanged=manifest["git_worktree_clean_after"]))
        print("TRAINING_RECHECKED seed=%d episodes=400 checkpoints=4" % seed, flush=True)
    for name, file in (("clean", "CLEAN_VICTIM_RESULTS.json"), ("oarl", "OARL_PROTOCOL_A_RESULTS.json")):
        historical = read(ROOT / "docs" / file)
        for run in historical["runs"]:
            manifest = read(ROOT / run["training_manifest"])
            require(manifest["status"] == "passed" and manifest["git_commit"] == run["training_commit"], "Historical training provenance changed")
            path = record((ROOT / run["training_manifest"]).parent / "episodes.jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            require(np.isclose(episode_windows(rows)[-1]["return_mean"], run["last100_training_return_mean"], rtol=0, atol=1e-12),
                    "Historical summary differs from raw training")
            groups[name][run["run_seed"]] = rows
    comparison = paired_training(groups)
    require(git("rev-parse", "HEAD") == initial_head and not git("status", "--porcelain"), "Analysis source changed")
    total = lambda field: sum(r["counts"][field] for r in results)
    result = dict(schema_version=1, kind="pgd_consistency_long_training_analysis", verified=True, gate_enabled=False,
        execution_commit=batch["git_commit"], analysis_commit=initial_head, generated_at_utc=datetime.datetime.utcnow().isoformat()+"Z",
        batch_directory=batch_directory.relative_to(ROOT).as_posix(), registry_sha256=batch["frozen_registry"]["sha256"],
        all_2000_episodes_completed=True, all_20_checkpoints_reverified=True, runs=sorted(results, key=lambda r:r["run_seed"]),
        aggregate=dict(episodes=2000, interactions=total("interactions"), updates=total("updates"), warmup_steps=total("warmup_steps"),
                       actor_training_forward_calls=sum(r["resources"]["phases"]["primary_update"]["network_forward_calls"]["actor"] for r in results),
                       pgd_gradient_evaluations=sum(r["defense_counts"]["gradient_evaluations"] for r in results),
                       concurrent_batch_wall_seconds=(datetime.datetime.fromisoformat(batch["finished_at_utc"]) -
                           datetime.datetime.strptime(batch["started_at_utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc)).total_seconds()),
        training_comparison=comparison, source_sha256=sources, analysis_command=[sys.executable, *sys.argv],
        audit_scope="Completed full raw box/KL/loss audits bound by fresh file hashes; streamed trajectories reaudited and 20 actors reloaded. No SUMO rerun.",
        limitation="Stochastic training outcomes and own-run early observation probes only; not frozen greedy evaluation or adversarial robustness. No hyperparameter or checkpoint reselection.")
    return result, registry, groups


def plot(result, groups, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    window = 20
    x = np.arange(window, 401)
    for seed in range(5):
        values = [r["episode_return"] for r in groups["pgd_consistency"][seed]]
        axes[0, 0].plot(x, np.convolve(values, np.ones(window)/window, "valid"), label="seed %d" % seed)
    for name in ("clean", "oarl", "pgd_consistency"):
        smooth = np.array([np.convolve([r["episode_return"] for r in groups[name][seed]], np.ones(window)/window, "valid") for seed in range(5)])
        line = axes[0, 1].plot(x, smooth.mean(0), label=name)[0]
        axes[0, 1].fill_between(x, smooth.mean(0)-smooth.std(0, ddof=1), smooth.mean(0)+smooth.std(0, ddof=1), color=line.get_color(), alpha=.14)
    for run in result["runs"]:
        label = "seed %d" % run["run_seed"]
        axes[1, 0].plot([100,200,300,400], [w["no_new_lane_request_action_fraction"] for w in run["transition_windows"]], marker="o", label=label)
        axes[1, 1].plot([100,200,300,400], [c["probe_profile"]["entropy_nats_mean"] for c in run["checkpoints"]], marker="o", label=label)
    titles = ("PGD: training return (20-episode mean)", "Training return: five-run mean +/- sample SD", "PGD: sampled action 2 fraction per 100 episodes", "PGD: entropy on 64 own-run early probes")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Completed training episode")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Episode return")
    axes[0, 1].set_ylabel("Episode return")
    axes[1, 0].set_ylim(-.03,1.03)
    axes[1, 1].set_ylabel("Entropy (nats)")
    fig.suptitle("Protocol A: descriptive training diagnostics; no frozen-policy defense test", fontsize=11)
    fig.savefig(str(output), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New ignored analysis directory")
    args = parser.parse_args()
    output = args.output.absolute()
    require(ROOT / ".local" in output.parents and not output.exists(), "Use a new ignored analysis directory")
    result, registry, groups = analyze(args.batch)
    output.mkdir(parents=True)
    write_json(output / "analysis.json", result)
    write_json(output / "frozen_defense_victims.json", registry)
    plot(result, groups, output / "training_diagnostics.png")
    print("ANALYSIS_PASSED=" + str(output), flush=True)
