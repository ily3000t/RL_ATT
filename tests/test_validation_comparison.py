import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_validation_comparison import aggregate_simple, core_cost_contrast, cost_rates, validate_simple_grid
sys.path.pop(0)


class ValidationComparisonTests(unittest.TestCase):
    def row(self, count, collided, converted, eligible, reward, steps):
        episodes = [dict(steps=steps, ego_collision_observed=i < collided,
                         attacked_steps=steps, changed_steps=steps, action_changed_steps=i,
                         linf_max=.2, l2_max=.5, scaled_linf_max=1.) for i in range(count)]
        return dict(attack="random", attack_seed=0, checkpoint_seed=0, episodes=count,
                    collisions=collided, episode_rows=episodes,
                    summary=dict(episode_return_mean=reward, steps=steps*count,
                                 attack_successes=converted, attack_success_eligible_episodes=eligible),
                    costs=dict(gradient_evaluations=0, policy_forward_calls=0, objective_evaluations=0,
                               new_shadow_transitions=0, shadow_steps=0, replay_steps=0, warmup_steps=0,
                               evaluator_policy_forward_calls=2*steps*count))

    def test_episode_weighting_keeps_clean_eligible_denominator_and_live_rates(self):
        a=self.row(2,1,1,1,10.,10)
        b=self.row(4,3,2,4,40.,5)
        b["attack_seed"]=1
        result=aggregate_simple([a,b],100.)
        self.assertEqual(result["mean_return"],30.)
        self.assertEqual(result["return_drop"],70.)
        self.assertEqual((result["eligible"],result["collision_conversions"],result["asr"]),(5,3,.6))
        self.assertEqual(result["collisions"],4)
        self.assertEqual(result["activity"]["action_change_rate"],7/40)

    def test_clean_repeats_cannot_inflate_grid_and_fgsm_cannot_gain_fake_seeds(self):
        rows=[dict(checkpoint_seed=s, attack_seed=seed, attack=name)
              for s in range(5) for seed in range(3) for name in ("none","random","pgd","oarl_bo")]
        rows += [dict(checkpoint_seed=s,attack_seed=0,attack="fgsm") for s in range(5)]
        validate_simple_grid(rows)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate_simple_grid(rows+[rows[0]])
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            validate_simple_grid(rows+[dict(checkpoint_seed=0,attack_seed=1,attack="fgsm")])
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            validate_simple_grid(rows[:-1])

    def test_clean_asr_control_excludes_its_own_collision_episodes(self):
        row=self.row(3,1,0,2,20.,2)
        row.update(attack="none")
        row["summary"].pop("attack_successes")
        row["summary"].pop("attack_success_eligible_episodes")
        for e in row["episode_rows"]:
            e["scaled_linf_max"]=None
        self.assertEqual(aggregate_simple([row],20.)["eligible"],2)
        self.assertEqual(aggregate_simple([row],20.)["asr"],0.)
        self.assertIsNone(aggregate_simple([row],20.)["activity"]["scaled_linf_max"])

    def test_core_costs_include_setup_and_do_not_normalize_away_early_termination(self):
        first=dict(episodes=2,live_steps=10,setup_shadow_steps=12,
                   costs=dict(gradient_evaluations=20,policy_forward_calls=40,new_shadow_transitions=5,shadow_steps=30))
        second=copy.deepcopy(first)
        second.update(live_steps=20)
        second["costs"].update(gradient_evaluations=10,policy_forward_calls=20,shadow_steps=20)
        result=core_cost_contrast(first,second)
        self.assertEqual(cost_rates(first)["per_episode"]["physical_shadow_steps_including_setup"],21.)
        self.assertEqual(cost_rates(first)["per_live_step"]["gradient_evaluations"],2.)
        self.assertEqual(result["per_episode_first_minus_second"]["gradient_evaluations"],5.)
        self.assertEqual(result["percent_first_minus_second"]["gradient_evaluations"],100.)
        self.assertEqual(result["per_episode_first_minus_second"]["physical_shadow_steps_including_setup"],5.)
        second["episodes"]=3
        with self.assertRaisesRegex(ValueError,"Unpaired"):
            core_cost_contrast(first,second)
