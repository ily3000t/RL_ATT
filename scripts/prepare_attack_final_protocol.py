"""Preregister the complete final matrix without launching or inspecting traffic."""

import copy
import hashlib
import json
import subprocess

from prepare_mechanism_controls import ROOT, source_sha256, sha256
from prepare_proposed_configs import protocol as seed_protocol, save
from rl_att.evaluation.configuration import validate_config

PROTOCOL = ROOT / "configs/research/attack_final_test.json"
METHOD_COMMIT = "d218309627830be924c48ef7fd9d984be11da247"
METHOD_DOCUMENT_COMMIT = "f0ada17"
METHOD_DOCUMENT = "docs/ATTACK_METHOD_FREEZE.md"
SEARCH_NAMES = ("none", "zero_one_budgeted_return", "zero_one_budgeted_safety",
                "ours_single_return", "ours_progress_return")
CAPS = (100, 200, 400)
CHECKPOINTS = tuple(range(5))
ATTACK_SEEDS = (0, 1, 2)
EPISODES = 50
STATISTICS = dict(
    primary=dict(first="ours_single_return", second="zero_one_budgeted_return",
                 endpoint="episode_return", direction="first_minus_second; negative is stronger",
                 budgets=list(CAPS)),
    secondary=dict(first="ours_single_return", second="ours_progress_return",
                   claim="Continuous effect/cost tradeoff; no equivalence or noninferiority margin"),
    averaging="attack seeds within checkpoint/traffic, then five fixed checkpoints within traffic",
    resampling_unit="traffic seed; jointly paired across methods, budgets and checkpoints",
    traffic_clusters=50, fixed_checkpoint_traffic_units=250,
    bootstrap=dict(draws=10000, seed=20261004, method="percentile_traffic_block",
                   descriptive_confidence=.95, primary_family_size=3,
                   primary_adjusted_confidence=1 - .05 / 3,
                   interpretation="Approximate conditional intervals, not a proof of significance"),
    fgsm="250 actual episodes; broadcast deterministic paired outcomes only, never costs or sample N",
    clean="250 canonical episodes from basic_attack0; other Clean episodes are integrity checks",
    diagnostics=["per_checkpoint", "per_traffic", "leave_one_traffic_out",
                 "wins_ties_losses_tolerance_1e-9", "clean_survived_collision_conversions"],
    costs=["gradient_evaluations", "attack_policy_forward_calls", "evaluator_policy_forward_calls",
           "total_policy_forward_calls", "new_shadow_transitions", "physical_shadow_steps_including_setup"],
    cost_denominators=["total", "mean_per_actual_episode", "per_live_step"],
    safety="Original SUMO collision definition; episode minimum TTC and TTC p5 / DRAC p95 with valid counts and nulls; do not present averaged episode percentiles as pooled percentiles",
    reporting="All 17 method/budget rows; negative cases and traffic concentration; no post-test model/config selection")
FAILURE_POLICY = dict(
    dispatch="Stop dispatch on any failed config or integrity audit; preserve every attempt",
    infrastructure_retries=1,
    retry="Only demonstrated infrastructure failure; same source/config/seeds/runtime/models, fresh output; retain failed attempt and cause; never retry because of poor scientific outcomes",
    invalidation="Seed/model/runtime/replay/budget/matrix mismatch invalidates the batch; do not drop traffic or replace it",
    exposure="Record exposure when any split-30 simulation starts. Algorithm changes after exposure require a new independent preregistered split; split 30 cannot become unseen again",
    debugging="Use split 10/20; no tuning on final outcomes")


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def template_paths(checkpoint, cap, attack_seed):
    if cap is None:
        return ["configs/evaluation/basic_validation_attack%d_seed%d.json" % (attack_seed, checkpoint)]
    return ["configs/evaluation/budget_validation_g%d_attack%d_seed%d.json" % (cap, attack_seed, checkpoint),
            "configs/evaluation/single_validation_g%d_attack%d_seed%d.json" % (cap, attack_seed, checkpoint)]


def final_config(checkpoint, cap, attack_seed):
    if checkpoint not in CHECKPOINTS or cap not in (None,) + CAPS or attack_seed not in ATTACK_SEEDS:
        raise ValueError("Use the complete preregistered final grid")
    sources = [read(p) for p in template_paths(checkpoint, cap, attack_seed)]
    result = copy.deepcopy(sources[0])
    if cap is not None:
        available = {a["name"]: a for s in sources for a in s["attacks"]}
        result["attacks"] = [copy.deepcopy(available[n]) for n in SEARCH_NAMES]
    result["episodes"] = EPISODES
    result["research_seeds"].update(split_id=30, episode_sumo_seeds=read(
        "configs/research_seed_splits.json")["splits"]["30"])
    return validate_config(result)


def matrix():
    configs, groups, templates = {}, [], set()
    for cap in (None,) + CAPS:
        for attack_seed in ATTACK_SEEDS:
            label = "basic" if cap is None else "search_g%d" % cap
            paths = []
            for checkpoint in CHECKPOINTS:
                path = "configs/evaluation/final_%s_attack%d_seed%d.json" % (label, attack_seed, checkpoint)
                configs[path] = final_config(checkpoint, cap, attack_seed)
                paths.append(path)
                templates.update(template_paths(checkpoint, cap, attack_seed))
            groups.append(dict(id="%s_attack%d" % (label, attack_seed), gradient_cap=cap,
                               attack_seed=attack_seed, configs=paths,
                               episodes=sum(c["episodes"] * len(c["attacks"]) for c in (configs[p] for p in paths))))
    return configs, groups, sorted(templates)


def make_protocol():
    if seed_protocol() != read("configs/research_seed_splits.json"):
        raise ValueError("Frozen traffic registry changed")
    subprocess.check_call(["git", "diff", "--exit-code", METHOD_COMMIT, "--", "main.py", "oarl.py",
                           "Environment", "Data", "rl_att", "requirements.txt"], cwd=str(ROOT))
    configs, groups, templates = matrix()
    old = read("configs/research/single_candidate_validation.json")
    sources = {p: source_sha256(p) for p in sorted(old["frozen_source_sha256"])}
    if sources != old["frozen_source_sha256"]:
        raise ValueError("Executed algorithm source differs from audited Single validation")
    victims = [v for v in read("configs/frozen_victims.json")["victims"] if v["victim"] == "clean"]
    if [v["run_seed"] for v in victims] != list(CHECKPOINTS):
        raise ValueError("Five predeclared Clean episode-400 models are required")
    training_proofs = []
    for victim in victims:
        checkpoint = ROOT / victim["checkpoint"]
        if sha256(checkpoint) != victim["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint file changed")
        manifest = checkpoint.parents[2] / "manifest.json"
        training_proofs.append(dict(path=manifest.relative_to(ROOT).as_posix(), sha256=sha256(manifest)))
    document_commit = subprocess.check_output(["git", "rev-parse", METHOD_DOCUMENT_COMMIT], cwd=str(ROOT)).decode().strip()
    document_blob = subprocess.check_output(["git", "show", document_commit + ":" + METHOD_DOCUMENT], cwd=str(ROOT))
    return dict(schema_version=1, kind="attack_final_test_preregistration", registration_date="2026-10-04",
                status="preregistered_not_run", final_outcomes_seen=False, final_execution_ready=False,
                frozen_method_commit=METHOD_COMMIT,
                method_document=dict(path=METHOD_DOCUMENT, commit=document_commit,
                                     git_blob_sha256=hashlib.sha256(document_blob).hexdigest()),
                frozen_source_sha256=sources, seed_protocol=read("configs/research_seed_splits.json"),
                seed_splits_sha256=sha256(ROOT / "configs/research_seed_splits.json"),
                frozen_victims_sha256=sha256(ROOT / "configs/frozen_victims.json"), victims=victims,
                runtime_references=training_proofs,
                zero_one_wheel=dict(path=".local/dependencies/zero-one/zoopt-0.4.2-py3-none-any.whl",
                    sha256="d015ab3633b8f1951c5caa59b05adff08bcd6655e6cc3a5f397b7abd2171faa5"),
                evidence=dict(path="docs/SINGLE_CANDIDATE_VALIDATION_RESULTS.json",
                              sha256=sha256(ROOT / "docs/SINGLE_CANDIDATE_VALIDATION_RESULTS.json")),
                templates={p: canonical_sha256(read(p)) for p in templates},
                config_digest_algorithm="sha256 canonical JSON UTF-8 sort_keys=True separators=(comma,colon) allow_nan=False",
                configs={p: canonical_sha256(c) for p, c in configs.items()}, groups=groups,
                search_methods=list(SEARCH_NAMES), checkpoint_seeds=list(CHECKPOINTS), attack_seeds=list(ATTACK_SEEDS),
                split_id=30, episodes=50, max_steps=200, gate_enabled=False,
                gradient_forward_caps=[[c, 2*c] for c in CAPS],
                common_shadow_limits=dict(new_shadow_transitions=200, shadow_steps=4000),
                totals=dict(configs=60, groups=12, raw_episodes=14500, attack_episodes=11500,
                            repeated_clean_episodes=3000, search_attack_episodes=9000,
                            simple_attack_episodes=2500, fgsm_episodes=250, canonical_clean_episodes=250,
                            unique_model_traffic_pairs=250, traffic_clusters=50, table_rows=17),
                seed_roles=dict(checkpoint_training_seed="0..4 model identity, never evaluation traffic seed",
                                policy_seed="unused: frozen greedy_argmax inference",
                                environment_seed="SeedSequence([20260921,1,30,2**31-2]) mod (2**31-1); Python/NumPy/Torch/initial SUMO",
                                episode_sumo_seed="Frozen 50-element split 30 list in order",
                                attack_seed="0,1,2 isolated by existing attack implementation; FGSM/no_attack deterministic"),
                execution=dict(parallel_groups=2, jobs_per_group=5, max_live_evaluations=10, policy_threads=1,
                               clean_reference_group="basic_attack0", require_clean_reference_before_other_groups=True,
                               provenance="Record actual execution SHA, clean tracked tree, config, role seeds, launch command, versions, checkpoint hashes; keep fixed for entire batch"),
                statistics=STATISTICS, failure_policy=FAILURE_POLICY,
                readiness_requirements=["Final dispatcher and 50-episode auditor committed/tested before exposure",
                                        "Whole-matrix completion, paired Clean regression, action/replay and all actual-cost ledger audits",
                                        "Raw episode identity validation and registered traffic-block statistics",
                                        "Read-only preflight passed on a clean exact execution commit"],
                scope="Freeze only. Existing validation auditors are not final-50 auditors; no automatic launch, new attack, defense, Gate or final tuning.")


def validate_registration(record, configs):
    expected_configs, groups, _ = matrix()
    if set(configs) != set(expected_configs):
        raise ValueError("Incomplete or unexpected final config matrix")
    for path, expected in expected_configs.items():
        if configs[path] != expected or record["configs"].get(path) != canonical_sha256(expected):
            raise ValueError("Final config differs from preregistered template: " + path)
    if len(record["configs"]) != 60 or record["groups"] != groups:
        raise ValueError("Duplicate, missing or altered final group")
    if record["statistics"] != STATISTICS or record["failure_policy"] != FAILURE_POLICY:
        raise ValueError("Registered analysis/failure policy changed")
    if record["split_id"] != 30 or record["gate_enabled"] or record["final_outcomes_seen"]:
        raise ValueError("This is a preregistration, not an exposed final result")
    return configs


def prepare():
    record = make_protocol()
    configs, _, _ = matrix()
    validate_registration(record, configs)
    for path, config in configs.items():
        save(ROOT / path, config)
    save(PROTOCOL, record)
    print("FINAL_PROTOCOL configs=60 groups=12 planned_episodes=14500 traffic_clusters=50 simulations=0")


if __name__ == "__main__":
    prepare()
