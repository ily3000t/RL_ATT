"""Post-step longitudinal safety measures from true SUMO state, not observations."""

import math
import numpy as np


def following_measures(gap_m, follower_speed, leader_speed):
    values = (gap_m, follower_speed, leader_speed)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite SUMO kinematics")
    closing = follower_speed - leader_speed
    row = {"gap_m": float(gap_m), "closing_speed_mps": float(closing)}
    if gap_m <= 0:
        row.update(ttc_s=0.0, drac_mps2=None, status="overlap_drac_undefined")
    elif closing <= 0:
        row.update(ttc_s=None, drac_mps2=0.0, status="not_closing")
    else:
        row.update(ttc_s=float(gap_m / closing), drac_mps2=float(closing ** 2 / (2 * gap_m)),
                   status="closing")
    return row


class SUMOMetrics:
    def __init__(self, connection, ego="Auto", lookahead_m=1000.0):
        if not math.isfinite(lookahead_m) or lookahead_m <= 0:
            raise ValueError("Safety search distance must be positive")
        self.connection, self.ego, self.lookahead_m = connection, ego, lookahead_m

    def sample(self):
        c, ego = self.connection, self.ego
        row = {"ego_collision_observed": ego in c.simulation.getCollidingVehiclesIDList(),
               "ego_present": ego in c.vehicle.getIDList(), "pairs": []}
        if not row["ego_present"]:
            return row
        lane = c.vehicle.getLaneID(ego)
        for relation, found in (("front", c.vehicle.getLeader(ego, self.lookahead_m)),
                                ("rear", c.vehicle.getFollower(ego, self.lookahead_m))):
            if not found or not found[0] or c.vehicle.getLaneID(found[0]) != lane:
                continue
            other, traci_gap = found
            follower, leader = (ego, other) if relation == "front" else (other, ego)
            # TraCI excludes the follower's minGap. Restore bumper-to-bumper space gap.
            gap = traci_gap + c.vehicle.getMinGap(follower)
            if gap > self.lookahead_m:
                continue
            pair = following_measures(gap, c.vehicle.getSpeed(follower), c.vehicle.getSpeed(leader))
            pair.update(relation=relation, follower=follower, leader=leader,
                        traci_gap_m=float(traci_gap))
            row["pairs"].append(pair)
        return row


def summarize_safety(samples, ttc_percentile=5, drac_percentile=95):
    pairs = [pair for sample in samples for pair in sample["pairs"]]
    ttc = [p["ttc_s"] for p in pairs if p["ttc_s"] is not None]
    drac = [p["drac_mps2"] for p in pairs if p["drac_mps2"] is not None]
    return {"minimum_ttc_s": min(ttc) if ttc else None,
            "ttc_low_percentile_s": float(np.percentile(ttc, ttc_percentile)) if ttc else None,
            "drac_high_percentile_mps2": float(np.percentile(drac, drac_percentile)) if drac else None,
            "ttc_percentile": ttc_percentile, "drac_percentile": drac_percentile,
            "sampled_steps": len(samples), "ego_present_steps": sum(s["ego_present"] for s in samples),
            "pair_samples": len(pairs), "finite_ttc_samples": len(ttc), "finite_drac_samples": len(drac),
            "not_closing_samples": sum(p["status"] == "not_closing" for p in pairs),
            "overlap_samples": sum(p["status"] == "overlap_drac_undefined" for p in pairs),
            "scope": "same_lane_front_and_rear_post_interaction_step"}
