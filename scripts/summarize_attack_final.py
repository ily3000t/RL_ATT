"""Export the complete registered final comparison, with paired traffic blocks."""

import argparse
import json
from pathlib import Path
import numpy as np
from prepare_attack_final_protocol import ROOT, PROTOCOL, CAPS
from prepare_mechanism_controls import sha256
from summarize_proposed_smoke import require
from final_statistics import paired_summary, outcome_cube
from run_baseline import git


def validate_reports(reports, protocol, commit):
    require(len(reports) == 12 and [r["group_id"] for r in reports] == [g["id"] for g in protocol["groups"]], "Incomplete/reordered final grid")
    conditions = set()
    for report, group in zip(reports, protocol["groups"]):
        require(report["kind"] == "attack_final_group" and report["verified"] and report["git_commit"] == commit and
                report["protocol_sha256"] == sha256(PROTOCOL) and report["split_id"] == 30 and
                report["gradient_cap"] == group["gradient_cap"] and report["attack_seed"] == group["attack_seed"] and
                report["episodes"] == group["episodes"] and report["clean_regression"]["passed"], "Wrong final group identity/provenance")
        names = [a["name"] for a in json.loads((ROOT / group["configs"][0]).read_text())["attacks"]]
        require(len(report["rows"]) == 5 * len(names) and
                {(r["checkpoint_seed"], r["attack"]) for r in report["rows"]} == {(m, n) for m in range(5) for n in names}, "Duplicate/missing model/method")
        for row in report["rows"]:
            key = report["gradient_cap"], report["attack_seed"], row["checkpoint_seed"], row["attack"]
            require(key not in conditions and row["episodes"] == 50, "Duplicate final condition")
            conditions.add(key)
            victim = next(v for v in protocol["victims"] if v["run_seed"] == row["checkpoint_seed"])
            require(row["checkpoint_sha256"] == victim["checkpoint_sha256"], "Wrong final model hash")
            ep = row["episode_rows"]
            require(len(ep) == 50 and [e["episode"] for e in ep] == list(range(1, 51)) and
                    [e["sumo_seed"] for e in ep] == protocol["seed_protocol"]["splits"]["30"], "Missing/unpaired final episodes")
    require(sum(r["episodes"] for r in reports) == 14500, "Wrong complete episode count")


def aggregate(rows):
    episodes = [ep for row in rows for ep in row["episode_rows"]]
    steps = sum(e["steps"] for e in episodes)
    costs = {k: sum(r["costs"][k] for r in rows) for k in rows[0]["costs"]}
    eligible = sum(r["summary"].get("attack_success_eligible_episodes", 0) for r in rows)
    successes = sum(r["summary"].get("attack_successes", 0) for r in rows)
    safety = {}
    for key in ("minimum_ttc_s", "ttc_low_percentile_s", "drac_high_percentile_mps2"):
        valid = [e["safety"][key] for e in episodes if e["safety"][key] is not None]
        safety[key] = dict(mean_of_episode_values=float(np.mean(valid)) if valid else None,
                           valid_episodes=len(valid), total_episodes=len(episodes))
    safety["scope"] = "Episode statistics, conditional on valid values; not pooled percentiles or transverse safety"
    return dict(actual_episodes=len(episodes), live_steps=steps,
                episode_return_mean=float(np.mean([e["episode_return"] for e in episodes])),
                return_drop_mean=float(np.mean([r["summary"]["return_drop_mean"] for r in rows])) if rows[0]["attack"] != "none" else 0.,
                collision_rate=sum(e["ego_collision_observed"] for e in episodes) / len(episodes),
                attack_success_rate=successes / eligible if eligible else None,
                attack_successes=successes, attack_success_eligible_episodes=eligible,
                attack_rate=sum(e["attacked_steps"] for e in episodes) / steps,
                action_change_rate=sum(e["action_changed_steps"] for e in episodes) / steps,
                observation_changed_rate=sum(e["changed_steps"] for e in episodes) / steps,
                linf_max=max(e["linf_max"] for e in episodes), l2_max=max(e["l2_max"] for e in episodes),
                scaled_linf_max=max(e["scaled_linf_max"] or 0. for e in episodes), safety=safety,
                costs=dict(total=costs, mean_per_actual_episode={k: v / len(episodes) for k, v in costs.items()},
                           per_live_step={k: v / steps for k, v in costs.items()}),
                attack_wall_seconds=sum(e["attack_wall_seconds"] for e in episodes))


def outcomes(rows, field):
    return [dict(checkpoint_seed=r["checkpoint_seed"], sumo_seed=e["sumo_seed"], attack_seed=r["attack_seed"], value=e[field])
            for r in rows for e in r["episode_rows"]]


def contrast(first, second, clean, traffic, second_deterministic=False):
    result = paired_summary(outcomes(first, "episode_return"), outcomes(second, "episode_return"), traffic,
                            second_deterministic=second_deterministic)
    a = outcome_cube(outcomes(first, "ego_collision_observed"), traffic)
    b = outcome_cube(outcomes(second, "ego_collision_observed"), traffic, second_deterministic)
    baseline = outcome_cube(outcomes(clean, "ego_collision_observed"), traffic, True)
    eligible = baseline == 0
    result["collision_conversion"] = dict(
        eligible_unique_model_traffic_pairs=int(eligible[:, :, 0].sum()),
        first_only_equivalent_units=float(((a > b) & eligible).mean(axis=2).sum()),
        second_only_equivalent_units=float(((b > a) & eligible).mean(axis=2).sum()),
        net_equivalent_units=float(((a - b) * eligible).mean(axis=2).sum()),
        definition="Average three correlated attacks within each fixed model/traffic pair; units may be fractional, not independent events")
    result["cost_difference_per_actual_episode"] = {
        k: sum(r["costs"][k] for r in first) / sum(r["episodes"] for r in first) -
           sum(r["costs"][k] for r in second) / sum(r["episodes"] for r in second) for k in first[0]["costs"]}
    return result


def build_result(reports, protocol, commit):
    validate_reports(reports, protocol, commit)
    by_method = {}
    for report in reports:
        for row in report["rows"]:
            if row["attack"] == "none" and report["group_id"] != "basic_attack0":
                continue
            item = dict(row, attack_seed=report["attack_seed"])
            by_method.setdefault((report["gradient_cap"], row["attack"]), []).append(item)
    clean = by_method[(None, "none")]
    table = [dict(gradient_cap=cap, attack=name, **aggregate(rows)) for (cap, name), rows in by_method.items()]
    require(len(table) == 17, "Incomplete 17-row comparison")
    traffic = protocol["seed_protocol"]["splits"]["30"]
    contrasts = []
    for cap in CAPS:
        first = by_method[(cap, "ours_single_return")]
        for name in ("zero_one_budgeted_return", "ours_progress_return", "zero_one_budgeted_safety", "random", "fgsm", "pgd", "oarl_bo"):
            second = by_method[(cap if name in protocol["search_methods"] else None, name)]
            contrasts.append(dict(gradient_cap=cap, first="ours_single_return", second=name,
                                  role="primary" if name == "zero_one_budgeted_return" else "mechanism" if name == "ours_progress_return" else "supplement",
                                  **contrast(first, second, clean, traffic, name == "fgsm")))
    return dict(kind="attack_final_complete_comparison", verified=True, experiment_commit=commit,
                protocol_sha256=sha256(PROTOCOL), totals=protocol["totals"], table=table, contrasts=contrasts,
                statistics=protocol["statistics"], failure_policy=protocol["failure_policy"],
                limitations="50 traffic clusters, five fixed checkpoints, correlated attack repeats; same original scenario. Observation box is not physical plausibility. No guaranteed safety/efficiency/novelty claim.")


def summarize(pipeline_path):
    protocol = json.loads(PROTOCOL.read_text())
    pipeline = json.loads(pipeline_path.read_text())
    require(pipeline["kind"] == "attack_final_pipeline" and pipeline["status"] in ("groups_passed", "passed") and
            pipeline["verified_episodes"] == 14500 and pipeline["protocol_sha256"] == sha256(PROTOCOL), "Incomplete final pipeline")
    commit, reports, proofs = pipeline["git_commit"], [], []
    require(git("rev-parse", "HEAD") == commit and not git("status", "--porcelain"), "Finalize on the fixed clean execution commit")
    for group in pipeline["groups"]:
        path = Path(group["audit_path"])
        require(group["status"] == "passed" and sha256(path) == group["audit_sha256"], "Changed group audit")
        report = json.loads(path.read_text())
        for name, digest in report["source_sha256"].items():
            require(sha256(ROOT / name) == digest, "Changed raw final input: " + name)
        reports.append(report)
        proofs.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path)))
    result = build_result(reports, protocol, commit)
    result.update(inputs=proofs, pipeline=pipeline_path.relative_to(ROOT).as_posix(),
                  raw_episode_verified_steps=sum(r["raw_episode_verified_steps"] for r in reports))
    if 'restart' in pipeline:
        for name, digest in pipeline['restart']['source_sha256'].items():
            require(sha256(ROOT / name) == digest, 'Original restart evidence changed: ' + name)
        result['restart'] = pipeline['restart']
    return result


def markdown(result):
    lines = ["# 最终未见交通攻击比较", "", "固定执行 SHA：`%s`。完整矩阵审计通过，14,500 个原始 episode，50 个交通聚类、五个固定模型。" % result["experiment_commit"],
             "", "| 方法 | 梯度档 | 实际 episode | Return | 碰撞率 | ASR | 梯度/episode | 攻击 forward/episode | physical shadow/episode |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["table"]:
        cost = row["costs"]["mean_per_actual_episode"]
        lines.append("| %s | %s | %d | %.6f | %.4f | %s | %.2f | %.2f | %.2f |" %
                     (row["attack"], row["gradient_cap"] or "—", row["actual_episodes"], row["episode_return_mean"], row["collision_rate"],
                      "null" if row["attack_success_rate"] is None else "%.4f" % row["attack_success_rate"],
                      cost["gradient_evaluations"], cost["policy_forward_calls"], cost["shadow_steps"]))
    lines.extend(["", "## 预登记主比较：Single − 预算版 Zero-One Return", "", "| 档位 | 配对 Return 差 | 描述性 95% 区间 | 三预算调整区间 | 净碰撞转换等效模型/交通单位 |", "| --- | ---: | --- | --- | ---: |"])
    for c in result["contrasts"]:
        if c["role"] == "primary":
            lines.append("| %d | %.6f | %s | %s | %.3f |" % (c["gradient_cap"], c["mean_delta"], c["descriptive_interval"],
                         c["primary_family_adjusted_interval"], c["collision_conversion"]["net_equivalent_units"]))
    lines.extend(["", "所有机制/简单对照的配对差、实际成本、逐模型/交通和 leave-one-traffic-out 在同名 JSON 中保留。",
                  "FGSM 为 250 个实际 episode；确定性结果只在配对计算广播，成本与样本数不复制。安全 episode 百分位条件均值不等于 pooled 百分位。",
                  "", "区间条件于五个固定模型，三个攻击 seed 不是三倍独立交通。净转换可为分数单位。负例与交通集中性必须纳入论文讨论，不能仅凭均值主张稳定安全优势或全面计算优势。", ""])
    if 'restart' in result:
        lines[0] = '# 最终固定交通攻击比较'
        lines.extend(['用户授权对中断测试完整重跑；仅本轮完整矩阵进入结果，旧尝试单独保留。',
                      'split 30 已在原尝试中暴露，本轮不是新的独立交通样本；算法、预算、模型和seed未调参。', ''])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require((ROOT / ".local/runs").resolve() in args.output.resolve().parents and not args.output.exists(), "Use a new ignored summary output")
    result = summarize(args.pipeline.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print("FINAL_COMPARISON_VERIFIED episodes=14500 rows=17")
