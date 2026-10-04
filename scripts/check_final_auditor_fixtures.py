"""Verify final raw primitives on immutable, already-used split-20 trajectories."""

import argparse
import json
from pathlib import Path
import sys
from analyze_attack_final import Reader, SEARCH, audit_raw, audit_witnesses, audit_oracle, verify_prefix, verify_sources
from summarize_proposed_smoke import require, audit_steps
from prepare_mechanism_controls import ROOT, sha256
from run_baseline import git

FIXTURES = (
    (".local/runs/20261001T010850536764Z-attack-benchmark", "verified-basic-validation.json"),
    (".local/runs/20260925T095242143052Z-attack-benchmark", "verified-shared-validation.json"),
    (".local/runs/20261003T074927928223Z-attack-benchmark", "verified-candidate-validation.json"),
)


def check():
    reader, methods, episodes, steps, proofs = Reader(), set(), 0, 0, []
    for name, audit_file in FIXTURES:
        directory = ROOT / name
        # Resolve the budget auditor's historical name without inspecting outcomes.
        if not (directory / audit_file).exists() and "shared" in audit_file:
            candidates = list(directory.glob("verified-*.json"))
            require(len(candidates) == 1, "Ambiguous historical audit")
            audit_file = candidates[0].name
        prior = reader.json(directory / audit_file)
        require(prior["verified"], "Historical fixture was not verified")
        for path, digest in prior["source_sha256"].items():
            require(sha256(ROOT / path) == digest, "Historical fixture input changed: " + path)
        batch = reader.json(directory / "batch.json")
        for run in batch["runs"]:
            child = Path(run["run_dir"])
            manifest, evaluation = reader.json(child / "manifest.json"), reader.json(child / "evaluation.json")
            require(manifest["status"] == "passed" and manifest["git_commit"] == batch["git_commit"], "Fixture provenance differs")
            config, clean, clean_steps = manifest["config"], None, None
            require(config["episodes"] == 20 and config["research_seeds"]["split_id"] == 20, "Only already-used validation fixtures")
            verify_sources(manifest, manifest["source_sha256_before"], child)
            for entry in evaluation["runs"]:
                spec = entry["attack"]
                if spec["name"] not in SEARCH + ("none", "random", "fgsm", "pgd", "oarl_bo"):
                    continue
                result = child / entry["results_directory"]
                ep, raw = reader.json(result / "episodes.json"), reader.steps(result / "steps.jsonl")
                count, summary = audit_raw(raw, ep, config, spec, clean)
                require({k: v for k, v in entry["summary"].items() if k != "safety"} == summary, "Fixture summary differs")
                if spec["name"] == "none":
                    clean, clean_steps = ep, raw
                else:
                    verify_prefix(raw, clean_steps)
                if spec["name"] in SEARCH:
                    audit_oracle(reader, child, entry, raw, ep, manifest["source_sha256_before"])
                    audit_witnesses(raw, spec["name"], config["research_seeds"]["attack_seed"], spec["parameters"])
                methods.add(spec["name"])
                episodes += len(ep)
                steps += count
        proofs.append(dict(batch=name + "/batch.json", source_commit=batch["git_commit"], audit=name + "/" + audit_file))
    require(methods == set(SEARCH + ("none", "random", "fgsm", "pgd", "oarl_bo")), "Missing fixture method")
    return dict(kind="final_auditor_historical_fixtures", verified=True, git_commit=git("rev-parse", "HEAD"),
                command=[sys.executable] + sys.argv, methods=sorted(methods), episodes=episodes, steps=steps,
                split_id=20, new_simulations=0, proofs=proofs, source_sha256=reader.sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require((ROOT / ".local/runs").resolve() in args.output.resolve().parents and not args.output.exists(), "Use new ignored output")
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("FINAL_FIXTURES_PASSED episodes=%d steps=%d new_simulations=0" % (result["episodes"], result["steps"]))
