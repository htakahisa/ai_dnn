"""Reinforce learned GC defense and arbitrate observed combat facing."""
import os

from ghost_champions_v1_macro import GhostChampionsV1DefenderController
from ghost_champions_v2.geometry import route
from .analysis import AttackSiteAnalysis
from .tendency import summarize_tendency
from .deployment import DefensivePosts
from .combat import CombatFacing
from .retake import CoveredRetake


class GCDefenderMacro:
    def __init__(self, analysis=None):
        self.analysis = analysis or AttackSiteAnalysis(checkpoint=os.environ.get("GC_V2_DEFENDER_ANALYSIS_CHECKPOINT"))
        self.posts = DefensivePosts(self.analysis.scenario)
        self.combat = CombatFacing()
        self.retake = CoveredRetake()
        self._clear_plan()

    def _clear_plan(self):
        self.goals, self.roles, self.baseline_goals = {}, {}, {}
        self.search_goals, self.baseline_search_goals = {}, {}
        self.reinforcement = None
        self.setup_configured = self.search_configured = False
        self.phase = "watch"
        self.allocation = dict(A=2, Mid=1, B=2)
        self.allocation_source = "existing_gc"
        self.selected_site = None

    def reset_round(self, winner=None):
        self.analysis.finish_round(winner=winner, no_plant=self.analysis.public_site is None)
        self.analysis.reset_round()
        self.combat.reset_round()
        self.retake.reset_round()
        self._clear_plan()

    def configure_setup(self, planner, char, state):
        if self.setup_configured or planner is None:
            return
        if not planner.round_initialized:
            planner.initialize_round(state.get("chars", ()))
        self.setup_configured = True
        self.baseline_goals = dict(planner.assignments)
        self.goals = dict(self.baseline_goals)
        tendency = summarize_tendency(self.analysis.previous_rounds)
        if tendency["biased"]:
            side = tendency["preferred_site"]
            allies = [c for c in state.get("chars", ()) if c.team == char.team and c.is_alive]
            goals, player = self.posts.reinforce_setup(self.goals, allies, side,
                int(state.get("defender_setup_ticks_remaining", 20)))
            if player is not None:
                planner.assignments = goals
                self.goals, self.reinforcement = goals, player
                self.selected_site, self.allocation_source = side, "round_history"
        self.roles = self.posts.roles(self.goals)
        self.allocation = {s: sum(self.posts.side(p) == s for p in self.goals.values()) for s in ("A", "Mid", "B")}

    def configure_search(self, search, char, state):
        if self.search_configured or search is None or state.get("is_planted") or not char.is_alive:
            return
        # This is the first operation in GC Search itself. Calling it here
        # retains its original RNG order and lets the lower layer use the
        # modified intention immediately, before choosing an action.
        if self.reinforcement is not None:
            search._ensure_defense_assignment(char, state["grid"], state.get("chars", ()))
            self.baseline_search_goals = {name: tuple(points[0]) for name, points in search._assigned_positions.items() if points}
            self.search_goals = self.posts.keep_search_roles(search, self.goals, state["grid"])
        self.search_configured = True

    def coordinate_setup(self, char, state, base_result):
        self.phase = "setup"
        if str(char.name) == self.reinforcement and tuple(char.pos) != self.goals[self.reinforcement]:
            # Only the added player follows the short Setup route. Avoid
            # permanent anchors, wait behind moving teammates, and never
            # take a long detour because the corridor is temporarily busy.
            anchors = {tuple(p) for n, p in self.goals.items() if n != self.reinforcement}
            path = route(self.posts.setup_grid, tuple(char.pos), [self.goals[self.reinforcement]], anchors)
            occupied = {tuple(c.pos) for c in state.get("chars", ()) if c.team == char.team and c.is_alive}
            return list(path[1] if len(path) > 1 and path[1] not in occupied else char.pos)
        return base_result

    def coordinate(self, char, state, base_result, locked_facing=None):
        self.phase = "retake" if state.get("is_planted") else "history_deploy" if self.reinforcement else "watch"
        return self.retake.coordinate(char, state, self.combat.coordinate(char, state, base_result, locked_facing))

    def snapshot(self):
        return dict(phase=self.phase, analysis=self.analysis.snapshot(), roles=dict(self.roles), goals=dict(self.goals),
                    baseline_goals=dict(self.baseline_goals), reinforcement=self.reinforcement,
                    search_goals=dict(self.search_goals), baseline_search_goals=dict(self.baseline_search_goals),
                    allocation=dict(self.allocation), allocation_source=self.allocation_source,
                    selected_site=self.selected_site,
                    combat_facing_corrections=self.combat.corrections, covered_retake_commits=self.retake.commits)


class GhostChampionsV2DefenderController(GhostChampionsV1DefenderController):
    def __init__(self, greedy=True, *, analysis=None):
        self.macro = GCDefenderMacro(analysis)
        self._scores_before = None
        super().__init__(greedy=greedy)

    def reset_round(self):
        winner = None
        if self._scores_before is not None and getattr(self, "game", None) is not None:
            old_names, old_scores = self._scores_before
            scores = {getattr(self.game, "attacker_team_name", "A"): int(getattr(self.game, "attacker_wins", 0)),
                      getattr(self.game, "defender_team_name", "D"): int(getattr(self.game, "defender_wins", 0))}
            # Score counters swap with sides; compare the previous teams' names.
            if scores.get(old_names[0], old_scores[0]) > old_scores[0]:
                winner = "A"
            elif scores.get(old_names[1], old_scores[1]) > old_scores[1]:
                winner = "D"
        self.macro.reset_round(winner)
        self._scores_before = None
        super().reset_round()

    def decide_move(self, char, state):
        # Setup receives raw engine state, so analysis must never run here.
        if state.get("defender_setup_active"):
            self.macro.phase = "setup"
            self.macro.configure_setup(getattr(self, "setup_planner", None), char, state)
            return self.macro.coordinate_setup(char, state, super().decide_move(char, state))
        if self._scores_before is None:
            self._scores_before = ((getattr(self.game, "attacker_team_name", "A"), getattr(self.game, "defender_team_name", "D")),
                                  (int(getattr(self.game, "attacker_wins", 0)), int(getattr(self.game, "defender_wins", 0))))
        self.macro.analysis.observe(char, state, int(getattr(self.game, "current_round", 1)))
        self.macro.configure_search(getattr(self, "search", None), char, state)
        locked_facing = char.facing if getattr(char, "facing_forced_this_tick", False) else None
        base_result = super().decide_move(char, state)
        return self.macro.coordinate(char, state, base_result, locked_facing)

    def defender_snapshot(self):
        return self.macro.snapshot()
