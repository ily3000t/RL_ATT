"""Export all seventeen fixed validation rows and keep the search comparison separate."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_proposed_smoke import require
from analyze_proposed_development import pair_outcomes
from summarize_budget_validation import concentration, validate_grid
from summarize_budget_sweep import aggregate as aggregate_search


SIMPLE = ("none", "random", "fgsm", "pgd", "oarl_bo")
LABELS = dict(none="Clean", random="Random", fgsm="FGSM", pgd="PGD", oarl_bo="OARL-BO",
              zero_one_budgeted_return="Zero-One Return", zero_one_budgeted_safety="Zero-One Safety",
              ours_progress_return="ProgressRetry Return", ours_progress_safety="ProgressRetry Safety")


def validate_simple_grid(rows):
    keys = [(r["checkpoint_seed"], r["attack_seed"], r["attack"]) for r in rows]
    expected = {(s, seed, name) for s in range(5) for seed in range(3) for name in SIMPLE
                if name != "fgsm" or seed == 0}
    require(len(keys) == len(set(keys)), "Duplicate simple condition (FGSM must run once)")
    require(set(keys) == expected, "Incomplete simple grid or duplicated deterministic FGSM seed")


def cost_rates(row):
    costs = dict(row["costs"])
    costs["setup_shadow_steps"] = row["setup_shadow_steps"]
    costs["physical_shadow_steps_including_setup"] = costs["shadow_steps"] + row["setup_shadow_steps"]
    require(row["episodes"] > 0 and row["live_steps"] > 0, "Empty cost denominator")
    return dict(per_episode={k: v / row["episodes"] for k, v in costs.items()},
                per_live_step={k: v / row["live_steps"] for k, v in costs.items()})


def activity(rows):
    episodes = [e for r in rows for e in r["episode_rows"]]
    counts = {k: sum(e[k] for e in episodes)
              for k in ("steps", "attacked_steps", "changed_steps", "action_changed_steps")}
    return dict(counts,
                attack_rate=counts["attacked_steps"] / counts["steps"],
                observation_changed_rate=counts["changed_steps"] / counts["steps"],
                action_change_rate=counts["action_changed_steps"] / counts["steps"],
                **{k: max(e[k] for e in episodes) for k in ("linf_max", "l2_max", "scaled_linf_max")})


def core_cost_contrast(first, second):
    require(first["episodes"] == second["episodes"], "Unpaired core cost denominator")
    a, b = cost_rates(first)["per_episode"], cost_rates(second)["per_episode"]
    keys = ("gradient_evaluations", "policy_forward_calls", "new_shadow_transitions",
            "physical_shadow_steps_including_setup")
    return dict(per_episode_first_minus_second={k: a[k] - b[k] for k in keys},
                percent_first_minus_second={k: 100 * (a[k] - b[k]) / b[k] if b[k] else None for k in keys})


def aggregate_simple(rows, clean_mean):
    name = rows[0]["attack"]
    require(all(r["attack"] == name for r in rows), "Mixed simple attack rows")
    total = sum(r["episodes"] for r in rows)
    costs = {k: sum(r["costs"][k] for r in rows) for k in rows[0]["costs"]}
    eligible = sum(r["summary"].get("attack_success_eligible_episodes",
                    sum(not e["ego_collision_observed"] for e in r["episode_rows"])) for r in rows)
    conversions = sum(r["summary"].get("attack_successes", 0) for r in rows)
    mean = sum(r["summary"]["episode_return_mean"] * r["episodes"] for r in rows) / total
    return dict(attack=name, gradient_cap=None, episodes=total,
                attack_seeds=[0] if name in ("none", "fgsm") else [0, 1, 2],
                mean_return=mean, return_drop=clean_mean - mean, collisions=sum(r["collisions"] for r in rows),
                collision_conversions=conversions, eligible=eligible, asr=conversions / eligible if eligible else None,
                live_steps=sum(r["summary"]["steps"] for r in rows), costs=costs, setup_shadow_steps=0,
                access="online_policy_only", protocol="fixed_config",
                activity=activity(rows),
                checkpoints=[dict(checkpoint_seed=r["checkpoint_seed"], attack_seed=r["attack_seed"], summary=r["summary"], costs=r["costs"])
                             for r in rows])


def summarize(path):
    hashes = {}
    def digest(p):
        p = p.resolve()
        if p not in hashes:
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for part in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(part)
            hashes[p] = h.hexdigest()
        return hashes[p]
    def audit(p, expected):
        require(digest(p) == expected, "Frozen report changed: " + str(p))
        d = json.loads(p.read_text())
        require(d["verified"] and d["source_sha256"], "Unverified report or missing input hashes")
        for name, value in d["source_sha256"].items():
            require(digest(ROOT / name) == value, "Audited source changed: " + name)
        return d
    pipeline = json.loads(path.read_text())
    protocol_path = ROOT / "configs/research/basic_validation_controls.json"
    protocol = json.loads(protocol_path.read_text())
    require(pipeline["status"] == "passed" and pipeline["verified_episodes"] == 1300, "Simple controls incomplete")
    require(digest(protocol_path) == pipeline["protocol_sha256"], "Protocol changed")
    require([g["attack_seed"] for g in pipeline["groups"]] == [0, 1, 2], "Missing or duplicated attack seed")
    search_path = ROOT / protocol["search_summary"]
    require(digest(search_path) == protocol["search_summary_sha256"], "Search summary changed")
    search = json.loads(search_path.read_text())
    search_pipeline_path = ROOT / protocol["search_pipeline"]
    require(digest(search_pipeline_path) == protocol["search_pipeline_sha256"], "Search pipeline changed")
    search_pipeline = json.loads(search_pipeline_path.read_text())
    search_reports = [audit(Path(g["audit_path"]), g["audit_sha256"]) for g in search_pipeline["groups"]]
    require(len(search_reports) == 9 and search_pipeline["status"] == "passed", "Search grid incomplete")
    validate_grid(search_reports)
    require(all(d["kind"] == "shared_compute_validation" and d["research_split_id"] == 20 and
                d["git_commit"] == d["analysis_commit"] == search_pipeline["git_commit"] and
                d["clean_regression"]["passed"] for d in search_reports), "Search provenance mismatch")
    require(search["inputs"] == [dict(path=str(Path(g["audit_path"]).relative_to(ROOT)), sha256=g["audit_sha256"],
                batch=d["batch"], gradient_cap=d["gradient_cap"], attack_seed=d["attack_seed"])
                for g, d in zip(search_pipeline["groups"], search_reports)], "Search summary input mismatch")
    basics, inputs = [], []
    for group in pipeline["groups"]:
        require(group["status"] == "passed", "Failed simple batch")
        p = Path(group["audit_path"])
        d = audit(p, group["audit_sha256"])
        seed = group["attack_seed"]
        require(d["kind"] == "basic_validation_controls" and d["attack_seed"] == seed and
                d["git_commit"] == pipeline["git_commit"] and d["clean_regression"]["passed"], "Simple report mismatch")
        require(d["analysis_commit"] == pipeline["git_commit"] and
                d["raw_episode_verified_steps"] == group["raw_verified_steps"], "Simple audit provenance or step mismatch")
        names = ("none", "random", "fgsm", "pgd", "oarl_bo") if seed == 0 else ("none", "random", "pgd", "oarl_bo")
        require(len(d["rows"]) == len(names) * 5 and {(r["checkpoint_seed"], r["attack"]) for r in d["rows"]} ==
                {(s, n) for s in range(5) for n in names}, "Missing simple conditions")
        for row in d["rows"]:
            row["attack_seed"] = seed
            basics.append(row)
        inputs.append(dict(path=str(p.relative_to(ROOT)), sha256=digest(p)))
    require(sum(r["episodes"] for r in basics) == 1300, "Simple episode count changed")
    validate_simple_grid(basics)
    clean = {(r["checkpoint_seed"]): r for r in basics if r["attack"] == "none" and r["attack_seed"] == 0}
    models = {s: r["checkpoint_sha256"] for s, r in clean.items()}
    traffic = search_reports[0]["traffic"]["episode_sumo_seeds"]
    for row in basics + [r for d in search_reports for r in d["rows"]]:
        require(row["checkpoint_sha256"] == models[row["checkpoint_seed"]], "Different frozen model")
        require(row["episodes"] == len(row["episode_rows"]) == 20 and [e["sumo_seed"] for e in row["episode_rows"]] == traffic and
                [e["episode"] for e in row["episode_rows"]] == list(range(1, 21)),
                "Different validation traffic or missing episodes")
        if row["attack"] == "none":
            require([e["trajectory_sha256"] for e in row["episode_rows"]] ==
                    [e["trajectory_sha256"] for e in clean[row["checkpoint_seed"]]["episode_rows"]], "Clean pairing changed")
    clean_mean = sum(r["summary"]["episode_return_mean"] for r in clean.values()) / 5
    effect_rows = []
    for name in ("none", "random", "fgsm", "pgd", "oarl_bo"):
        rs = [r for r in basics if r["attack"] == name and (name != "none" or r["attack_seed"] == 0)]
        effect_rows.append(aggregate_simple(rs, clean_mean))
    for p in search["pooled"]:
        rs = [r for d in search_reports if d["gradient_cap"] == p["gradient_cap"] for r in d["rows"] if r["attack"] == p["attack"]]
        require(dict(gradient_cap=p["gradient_cap"], attack=p["attack"], **aggregate_search(rs)) == p,
                "Search aggregate differs from verified episodes")
        costs = dict(p["costs"], evaluator_policy_forward_calls=2*p["live_steps"],
                     objective_evaluations=sum(r["summary"]["objective_evaluations"] for r in rs))
        effect_rows.append(dict(attack=p["attack"], gradient_cap=p["gradient_cap"], episodes=p["episodes"], attack_seeds=[0, 1, 2],
            mean_return=p["mean_return"], return_drop=clean_mean-p["mean_return"], collisions=p["collisions"],
            collision_conversions=p["collision_conversions"], eligible=p["clean_noncollision_episodes"], asr=p["conversion_rate"],
            live_steps=p["live_steps"], costs=costs, activity=activity(rs),
            setup_shadow_steps=p["oracle_setup_shadow_steps"], access="white_box_and_sumo_oracle", protocol="shared_multiresource_caps",
            checkpoints=[dict(checkpoint_seed=r["checkpoint_seed"], attack_seed=d["attack_seed"],
                              summary=r["summary"], costs=r["audit"]["costs"])
                         for d in search_reports if d["gradient_cap"] == p["gradient_cap"]
                         for r in d["rows"] if r["attack"] == p["attack"]]))
    require(len(effect_rows) == 17, "Comparison must retain all seventeen rows")
    for row in effect_rows:
        row["collision_rate"] = row["collisions"] / row["episodes"]
        row["cost_rates"] = cost_rates(row)
    comparisons = []
    for p in search["pooled"]:
        for simple in ("random", "fgsm", "pgd", "oarl_bo"):
            counts, differences, discordances, pairs = {}, [], [], []
            for d in search_reports:
                if d["gradient_cap"] != p["gradient_cap"]:
                    continue
                for row in d["rows"]:
                    if row["attack"] != p["attack"]:
                        continue
                    s, seed = row["checkpoint_seed"], d["attack_seed"]
                    ref = next(b for b in basics if b["attack"] == simple and b["checkpoint_seed"] == s and
                               b["attack_seed"] == (0 if simple == "fgsm" else seed))
                    pair = pair_outcomes(clean[s]["episode_rows"], row["episode_rows"], ref["episode_rows"])
                    for k, v in pair.items():
                        if k not in ("episodes", "return_first_minus_second_mean"):
                            counts[k] = counts.get(k, 0) + v
                    differences.append(pair["return_first_minus_second_mean"])
                    pairs.append(dict(attack_seed=seed, checkpoint_seed=s, **pair))
                    discordances.extend(dict(checkpoint_seed=s, attack_seed=seed, **e) for e in pair["episodes"]
                                        if e["first_collision"] != e["second_collision"])
            comparisons.append(dict(search=p["attack"], gradient_cap=p["gradient_cap"], simple=simple,
                return_search_minus_simple=sum(differences)/len(differences), paired_records=300,
                fgsm_reference_reused=simple == "fgsm", collision_discordances=discordances,
                traffic_concentration=concentration(pairs, traffic), **counts))
    core = []
    for cap in (100, 200, 400):
        for objective in ("return", "safety"):
            pairs = [p for p in search["paired"] if p["gradient_cap"] == cap and p["objective"] == objective]
            require(len(pairs) == 15, "Core paired comparison incomplete")
            first, second = [next(r for r in effect_rows if r["gradient_cap"] == cap and r["attack"] == prefix + objective)
                             for prefix in ("ours_progress_", "zero_one_budgeted_")]
            core.append(dict(gradient_cap=cap, objective=objective,
                **{k:sum(p[k] for p in pairs) for k in ("first_only_conversion", "second_only_conversion", "both_collision", "neither_collision")},
                return_progress_minus_zero_one=sum(p["return_first_minus_second_mean"] for p in pairs)/len(pairs),
                costs=core_cost_contrast(first, second),
                leave_one_traffic_out_net_range=next(c["leave_one_traffic_out_net_range"] for c in search["concentration"]
                                                   if c["gradient_cap"]==cap and c["objective"]==objective)))
    result = dict(kind="validation_simple_and_search_comparison", research_split_id=20, rows=effect_rows,
        core_search_comparison=core, search_vs_simple=comparisons, inputs=inputs,
        search_summary=dict(path=str(search_path.relative_to(ROOT)), sha256=digest(search_path)),
        pipeline=dict(path=str(path.relative_to(ROOT)), sha256=digest(path)),
        protocol=dict(path=str(protocol_path.relative_to(ROOT)), sha256=digest(protocol_path)),
        traffic=traffic, frozen_checkpoint_sha256=models,
        basic_experiment_commit=pipeline["git_commit"], search_experiment_commit=search["experiment_commit"],
        new_episodes=1300, new_attack_episodes=1000, unique_clean_pairs=100, unique_fgsm_pairs=100,
        raw_verified_new_steps=sum(g["raw_verified_steps"] for g in pipeline["groups"]),
        limitation="Shared validation traffic; FGSM references reused only for pairing. Different objectives/access rights prevent causal attribution to simulation search.")
    result["summary_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit comparison script before publishing")
    return result


def markdown(result):
    lines = ["# 同验证交通：简单攻击与轨迹搜索比较", "", "完整17行；FGSM和Clean只有100个唯一条件，其余各300条相关记录。", "",
             "| 方法 | 梯度/Forward上限 | N | Return | Drop | 碰撞（%） | ASR转换/分母（%） |", "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for r in result["rows"]:
        budget = "%d/%d" % (r["gradient_cap"],2*r["gradient_cap"]) if r["gradient_cap"] else "固定配置"
        lines.append("| %s | %s | %d | %.4f | %.4f | %d/%d (%.2f) | %d/%d (%.2f) |" %
                     (LABELS[r["attack"]],budget,r["episodes"],r["mean_return"],r["return_drop"],r["collisions"],r["episodes"],
                      100*r["collision_rate"],r["collision_conversions"],r["eligible"],100*r["asr"]))
    lines += ["", "## 实际成本总量", "",
              "| 方法 | 上限 | 梯度 | 攻击Forward | 目标评估 | 新转移 | 规划Shadow | Replay | 规划Warmup | Setup Shadow |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in result["rows"]:
        c=r["costs"]
        lines.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d |" %
                     (LABELS[r["attack"]], r["gradient_cap"] or "固定", c["gradient_evaluations"],c["policy_forward_calls"],
                      c["objective_evaluations"],c["new_shadow_transitions"],c["shadow_steps"],c["replay_steps"],c["warmup_steps"],r["setup_shadow_steps"]))
    lines += ["", "## 按episode和真实交互步记账", "",
              "| 方法 | 上限 | 真实步 | 梯度/episode | Forward/episode | 全Shadow/episode | 梯度/真实步 | Forward/真实步 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in result["rows"]:
        a,b=r["cost_rates"]["per_episode"],r["cost_rates"]["per_live_step"]
        lines.append("| %s | %s | %d | %.2f | %.2f | %.2f | %.4f | %.4f |" %
                     (LABELS[r["attack"]],r["gradient_cap"] or "固定",r["live_steps"],a["gradient_evaluations"],a["policy_forward_calls"],
                      a["physical_shadow_steps_including_setup"],b["gradient_evaluations"],b["policy_forward_calls"]))
    lines += ["", "目标评估不是跨方法可互换的单位：BO为标量候选评估，搜索为候选轨迹评分；梯度与Forward单独计数。规划Shadow=新转移+Replay+规划Warmup；全Shadow另加Setup。公共评估器固定2次Forward/真实步，未混入攻击Forward；所有资源的每episode/每步值均在JSON中保存。实际episode成本包含提前终止的影响。", "",
              "## 核心搜索配对：当前方法减预算版Zero-One", "",
              "| 上限 | 目标 | 当前独有转换 | Zero-One独有转换 | 净差 | 回报差 | 去掉一交通净差范围 | 梯度/episode差 | Forward/episode差 | 全Shadow/episode差 |", "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for r in result["core_search_comparison"]:
        a,b=r["first_only_conversion"],r["second_only_conversion"]
        c=r["costs"]["per_episode_first_minus_second"]
        lines.append("| %d/%d | %s | %d | %d | %+d | %.4f | %s | %+.2f | %+.2f | %+.2f |" %
                     (r["gradient_cap"],2*r["gradient_cap"],r["objective"],a,b,a-b,r["return_progress_minus_zero_one"],
                      r["leave_one_traffic_out_net_range"],c["gradient_evaluations"],c["policy_forward_calls"],c["physical_shadow_steps_including_setup"]))
    lines += ["", "## 搜索与FGSM的同交通配对", "",
              "FGSM是本轮简单方法中ASR最高者。下表将其100个确定性参考复用于搜索的三个attack seed，形成300条相关配对；没有新增FGSM样本或独立性。其他三种简单攻击的48组完整配对计数见JSON。", "",
              "| 搜索 | 上限 | 搜索独有转换 | FGSM独有转换 | 净差/261 | 回报差 | 去掉一交通净差范围 |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for r in result["search_vs_simple"]:
        if r["simple"] != "fgsm":
            continue
        a,b=r["first_only_conversion"],r["second_only_conversion"]
        lines.append("| %s | %d/%d | %d | %d | %+d | %.4f | %s |" %
                     (LABELS[r["search"]],r["gradient_cap"],2*r["gradient_cap"],a,b,a-b,r["return_search_minus_simple"],
                      r["traffic_concentration"]["leave_one_traffic_out_net_range"]))
    lines += ["", "## 扰动与执行率", "",
              "| 方法 | 上限 | Attack % | 观测改变 % | 动作改变 % | 最大L_inf | 最大L2 | 最大归一化L_inf |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in result["rows"]:
        a=r["activity"]
        lines.append("| %s | %s | %.2f | %.2f | %.2f | %.6f | %.6f | %.6f |" %
                     (LABELS[r["attack"]],r["gradient_cap"] or "固定",100*a["attack_rate"],100*a["observation_changed_rate"],
                      100*a["action_change_rate"],a["linf_max"],a["l2_max"],a["scaled_linf_max"]))
    lines += ["", "逐checkpoint/attack seed的Return、ASR及TTC/DRAC原始小型摘要保留在JSON checkpoints中；TTC是同车道前后车在交互后采样，缺少闭合样本时为null。没有把分组分位数平均成总体分位数，也没有把轨迹搜索内部风险分数当成真实安全指标。SUMO碰撞仍包含原minGap语义。", "",
              "所有费用按实际记录。OARL-BO使用共同扰动盒内共享仿射子集，并沿用在线obs2=obs1适配。Random无需策略梯度；FGSM/PGD需要白盒策略；BO保留原actor/critic目标；轨迹搜索额外拥有SUMO oracle权限。不同目标与权限的比较不能单独归因于搜索模块。完整配对得失、逐模型指标与来源哈希见机器摘要。", ""]
    return "\n".join(lines)


def plot(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (cost, title) in enumerate((("gradient_evaluations", "Gradient evaluations"), ("policy_forward_calls", "Attack policy forwards"))):
        for prefix, label, color in (("zero_one_budgeted_", "Budgeted Zero-One", "#2864ad"), ("ours_progress_", "ProgressRetry", "#cc5928")):
            for objective, style in (("return", "o-"), ("safety", "x--")):
                rows = [r for r in result["rows"] if r["attack"] == prefix + objective]
                xs = [r["cost_rates"]["per_episode"][cost] for r in rows]
                for ax, values in ((axes[0,col], [100*r["asr"] for r in rows]), (axes[1,col], [r["mean_return"] for r in rows])):
                    ax.plot(xs, values, style, color=color, label=label + " " + objective, markersize=5)
                    if objective == "return":
                        for x,y,r in zip(xs,values,rows):
                            ax.annotate(str(r["gradient_cap"]), (x,y), xytext=(4,6), textcoords="offset points", fontsize=8, color=color)
        for name, color, marker in (("none", "#777777", "s"), ("random", "#659049", "D"), ("fgsm", "#8c56a4", "^"), ("pgd", "#111111", "v"), ("oarl_bo", "#c4a228", "P")):
            r=next(r for r in result["rows"] if r["attack"] == name)
            x=r["cost_rates"]["per_episode"][cost]
            for ax,y in ((axes[0,col],100*r["asr"]), (axes[1,col],r["mean_return"])):
                ax.scatter([x],[y],color=color,marker=marker,label=LABELS[name],s=40,zorder=5)
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
            ax.set_xlim(left=-120)
        axes[0,col].set_title(title + " / episode")
        axes[1,col].set_xlabel(title + " / episode")
    axes[0,0].set_ylabel("Collision conversion ASR (%)")
    axes[1,0].set_ylabel("Mean episode return")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.965),ncol=5,fontsize=8)
    fig.suptitle("Fixed simple controls and all three search budgets on validation traffic", fontsize=13)
    fig.text(.5,.016,"Search additionally uses SUMO oracle (see actual shadow costs). FGSM/Clean: 100 unique conditions; others: 300 correlated records.\nLabels: per-block gradient cap; forward cap = 2x. Early termination affects realized episode costs. No causal attribution or independence claim.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.075,1,.87))
    fig.savefig(str(directory / "validation-simple-search-cost.png"),dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    result = summarize(args.pipeline.resolve())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    args.markdown.write_text(markdown(result),encoding="utf-8")
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        plot(result, args.figures)
    print("VALIDATION_COMPARISON_PASSED rows=17 new_attacks=1000")
