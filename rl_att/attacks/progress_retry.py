"""Development revision: stop further restarts after a retry makes no progress.

This is a compute allocation heuristic, not an unreachability certificate.
The first independent retry remains available and all original budgets apply.
"""

from .behavior_search import BehaviorSearch
from .proposed import ProposedAttack


class ProgressRetrySearch(BehaviorSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_retry = {}
        self.suppressed = {}

    def invert(self, node, history, target, attempt):
        previous_best = node["best_margin"][target]
        before = len(self.attempt_trace)
        item = super().invert(node, history, target, attempt)
        if attempt > 0 and len(self.attempt_trace) > before:
            margin = self.attempt_trace[-1]["margin"]
            self.last_retry[(history, target)] = dict(previous_best=previous_best, margin=margin,
                                                      improved=margin > previous_best)
        return item

    def retry_targets(self, node, history, targets):
        allowed = []
        for target in super().retry_targets(node, history, targets):
            decision = self.last_retry.get((history, target))
            if node["attempts"][target] <= 1 or decision is None or decision["improved"]:
                allowed.append(target)
            else:
                key = (history, target, node["attempts"][target])
                self.suppressed[key] = dict(history=list(history), target=target,
                                             attempts_so_far=node["attempts"][target],
                                             reason="retry_did_not_improve_best_margin", **decision)
        return allowed

    def metadata(self):
        metadata = super().metadata()
        metadata.update(retry_rule="strict_margin_progress", suppressed_retry_targets=len(self.suppressed),
                        retry_stop_trace=list(self.suppressed.values()))
        return metadata


class ProgressRetryAttack(ProposedAttack):
    engine_type = ProgressRetrySearch
    method_prefix = "ours_progress_"

    def __init__(self, retry_rule="strict_margin_progress", **kwargs):
        if retry_rule != "strict_margin_progress":
            raise ValueError("Explicit strict margin progress retry rule required")
        super().__init__(**kwargs)

    def __call__(self, observation, victim, context):
        result = super().__call__(observation, victim, context)
        result.metadata["retry_rule"] = "strict_margin_progress"
        return result
