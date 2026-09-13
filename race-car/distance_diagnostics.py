"""Passive distance accounting, not avoidability or a same-traffic counterfactual.

For initial speed v0 and budget N the unclipped always-accelerate ceiling is
v0*N + .05*N*(N+1). At executed tick t (1-based), attribute
(.1 - actual_delta_vx)*(N-t+1) to that action and its budget third. Actual
speed deltas include braking at zero (and any other speed clipping).
After T executed ticks, termination loss is the sum of the N-T hypothetical
speeds final_vx + .1*j. Thus ceiling - actual_distance equals action opportunity
loss plus termination loss, up to floating-point roundoff. These are accounting
attributions, NOT claims that the lost distance was safely achievable.

Plan counters sample last_plan once per controller decision, not per action or
tick. Missing fields remain unreported. Short/zero selected safe_ticks do not
prove that every candidate was unsafe; recovery_checks does not imply vetoes.
Optional blocked/veto fields are summed as reported, never inferred. Target
switches compare adjacent decisions with reported targets, not steering or
physical lane reversals. No traffic, RNG, or controller mutation is used here.
"""

from collections import Counter


class DistanceDiagnostics:
    def __init__(self, initial_speed, max_ticks):
        self.initial_speed = float(initial_speed)
        self.max_ticks = max_ticks
        self.executed_ticks = 0
        self.final_speed = self.initial_speed
        self.action_counts = Counter()
        self.loss_by_action = Counter()
        self.phases = {name: {"action_counts": Counter(), "loss_by_action": Counter()}
                       for name in ("early", "middle", "late")}
        self.decision_counts = Counter(decisions=0, decisions_with_plan=0)
        self.previous_target = None

    def record_tick(self, action, previous_speed, actual_speed):
        """Record an executed action only; phases use floor(3*(t-1)/N)."""
        phase = ("early", "middle", "late")[min(2, 3 * self.executed_ticks // self.max_ticks)]
        loss = (0.1 - (actual_speed - previous_speed)) * (self.max_ticks - self.executed_ticks)
        self.action_counts[action] += 1
        self.loss_by_action[action] += loss
        self.phases[phase]["action_counts"][action] += 1
        self.phases[phase]["loss_by_action"][action] += loss
        self.executed_ticks += 1
        self.final_speed = float(actual_speed)

    def record_decision(self, plan=None):
        """Decision-level reporting only, not explanations of tick safety."""
        counts = self.decision_counts
        counts["decisions"] += 1
        plan = plan or {}
        counts["decisions_with_plan"] += bool(plan)
        for field in ("recovery_checks", "recovery_vetoes", "observed_blocked", "unseen_blocked"):
            if field in plan:
                counts[f"{field}_reported_decisions"] += 1
                counts[f"{field}_sum"] += int(plan[field])
                counts[f"{field}_positive_decisions"] += plan[field] > 0
        if "recovery_safe" in plan:
            counts["recovery_safe_reported_decisions"] += 1
            counts["recovery_safe_decisions"] += bool(plan["recovery_safe"])
            counts["recovery_unsafe_decisions"] += not plan["recovery_safe"]
        if "safe_ticks" in plan:
            counts["safe_ticks_reported_decisions"] += 1
            counts["selected_zero_safe_ticks_decisions"] += plan["safe_ticks"] == 0
        if "horizon" in plan:
            counts["horizon_reported_decisions"] += 1
        if "safe_ticks" in plan and "horizon" in plan:
            counts["safe_horizon_comparable_decisions"] += 1
            counts["selected_short_horizon_decisions"] += plan["safe_ticks"] < plan["horizon"]
        target = plan.get("target")
        if target is not None:
            counts["target_reported_decisions"] += 1
            if self.previous_target is not None:
                counts["target_comparable_decisions"] += 1
                counts["target_switch_decisions"] += target != self.previous_target
        self.previous_target = target

    def result(self, actual_distance):
        """Keep unrounded distances so the identity can be checked in JSON."""
        remaining = self.max_ticks - self.executed_ticks
        ceiling = self.initial_speed * self.max_ticks + 0.05 * self.max_ticks * (self.max_ticks + 1)
        opportunity = sum(self.loss_by_action.values())
        termination = self.final_speed * remaining + 0.05 * remaining * (remaining + 1)
        distance_loss = ceiling - actual_distance
        return {
            "attribution": "diagnostic_only_not_avoidability_or_same_traffic_counterfactual",
            "initial_speed": self.initial_speed, "max_ticks": self.max_ticks,
            "executed_ticks": self.executed_ticks, "remaining_ticks": remaining,
            "ceiling": ceiling, "actual_distance": actual_distance,
            "distance_loss": distance_loss, "action_opportunity_loss": opportunity,
            "termination_loss": termination,
            "identity_residual": distance_loss - opportunity - termination,
            "action_counts": dict(self.action_counts), "loss_by_action": dict(self.loss_by_action),
            "phases": {name: {"action_counts": dict(group["action_counts"]),
                              "loss_by_action": dict(group["loss_by_action"])}
                       for name, group in self.phases.items()},
            "decision_counts": dict(self.decision_counts),
        }