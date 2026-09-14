"""Export a small, checked Stage 2 validation summary without raw trajectories."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def checked_run(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
    if manifest["status"] != "passed" or manifest["git_commit"] != result["git_commit"]:
        raise ValueError("Evaluation has not passed with the recorded code")
    if any(name != "Data/StraightRoad.sumocfg" for name in manifest["changed_source_files"]):
        raise ValueError("Unexpected source change during evaluation")
    references = {(r["victim"], r["run_seed"]): r for r in manifest["victim_references"]}
    for run in result["runs"]:
        ref = references[(run["victim"], run["run_seed"])]
        if not run["frozen_unchanged"] or run["checkpoint_sha256"] != ref["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint validation missing")
        if hashlib.sha256((ROOT / ref["checkpoint"]).read_bytes()).hexdigest() != ref["checkpoint_sha256"]:
            raise ValueError("Checkpoint file changed since evaluation")
    return manifest, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, equivalence = checked_run(args.equivalence)
    smoke_manifest, smoke = checked_run(args.smoke)
    expected = {(v, s) for v in ("clean", "oarl") for s in range(5)}
    if len(equivalence["runs"]) != 10 or {(r["victim"], r["run_seed"]) for r in equivalence["runs"]} != expected:
        raise ValueError("Require all ten frozen victims")
    for run in equivalence["runs"]:
        if (run["attack"]["name"] != "none" or not run["legacy_equivalence"]["passed"]
                or run["legacy_equivalence"]["episodes"] != 20 or run["summary"]["attack_rate"] != 0):
            raise ValueError("Full legacy NoAttack equivalence required")
    if len(smoke["runs"]) != 4 or {(r["victim"], r["run_seed"], r["attack"]["name"]) for r in smoke["runs"]} != {
            (v, 0, a) for v in ("clean", "oarl") for a in ("none", "oarl_bo")}:
        raise ValueError("Require paired smoke checks for both victims")
    for run in smoke["runs"]:
        rows = json.loads((args.smoke / run["results_directory"] / "episodes.json").read_text(encoding="utf-8"))
        if len(rows) != 2 or run["summary"]["steps"] != 40:
            raise ValueError("Smoke must complete two 20-step episodes")
        if run["attack"]["name"] == "oarl_bo":
            if run["summary"]["objective_evaluations"] != 200 or run["summary"]["attack_rate"] != 1:
                raise ValueError("Smoke BO budget was not fully executed")
            for line in (args.smoke / run["results_directory"] / "steps.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                metadata = row["attack_metadata"]
                if len(metadata["trace"]) != 5 or metadata["objective_value"] != max(t["objective"] for t in metadata["trace"]):
                    raise ValueError("Recorded BO winner or evaluations inconsistent")
                u1, u2 = metadata["parameters"]["u1"], metadata["parameters"]["u2"]
                if not 0.8 <= u1 <= 1.2 or not -0.05 <= u2 <= 0.05:
                    raise ValueError("Recorded BO parameter outside budget")
                if any(abs(d) > 0.2 * abs(o) + 0.05 + 2e-7 for d, o in zip(row["perturbation"], row["observation"])):
                    raise ValueError("Recorded perturbation outside affine envelope")

    def compact(run):
        return {key: run[key] for key in ("victim", "run_seed", "attack", "checkpoint_sha256", "frozen_unchanged", "summary")}

    summary = {"stage": 2, "scope": "Adapter equivalence and safety instrumentation; not an attack efficacy benchmark",
               "equivalence": {"directory": args.equivalence.resolve().relative_to(ROOT).as_posix(),
                               "git_commit": manifest["git_commit"], "config_file": manifest["config_file"],
                               "all_10_victims_matched": True, "episodes": 200,
                               "steps": sum(r["summary"]["steps"] for r in equivalence["runs"]),
                               "runs": [compact(r) for r in equivalence["runs"]]},
               "bo_smoke": {"directory": args.smoke.resolve().relative_to(ROOT).as_posix(),
                            "git_commit": smoke_manifest["git_commit"], "config_file": smoke_manifest["config_file"],
                            "all_checks_passed": True, "runs": [compact(r) for r in smoke["runs"]]}}
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("STAGE2_VALIDATION=" + str(args.output))


if __name__ == "__main__":
    main()
