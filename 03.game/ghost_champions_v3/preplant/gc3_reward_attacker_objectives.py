"""Reward actual micro objectives without completion resets or cancel loops."""
from collections import Counter, defaultdict

from game_core import PLANT_REQUIRED_TICKS
from grid_paths import distance_map

SPIKE_PICKUP_REWARD = 3.0
PLANT_PROGRESS_REWARD_PER_TICK = 0.25
PLANT_CANCEL_EXTRA_PENALTY = 0.05
PLANT_COMPLETE_REWARD = 5.0


def objective_reward_settings():
    return dict(SPIKE_PICKUP_REWARD=SPIKE_PICKUP_REWARD,
                PLANT_PROGRESS_REWARD_PER_TICK=PLANT_PROGRESS_REWARD_PER_TICK,
                PLANT_CANCEL_EXTRA_PENALTY=PLANT_CANCEL_EXTRA_PENALTY,
                PLANT_COMPLETE_REWARD=PLANT_COMPLETE_REWARD,
                approach_reward="new best proximity only; no reward for creating/changing an objective",
                pickup_reward="personal, once per actual ground pickup, before shooting",
                plant_progress_reward="personal, actual valid channel advancement, including completion tick",
                plant_cancel="discount-compensated refund of this attempt's progress bonuses, plus extra penalty",
                plant_completion_reward="personal to the planter, once per actual plant; independent of the round result",
                retrieval_progress_reward="personal new-best approach for each actor; bounded per real drop")


class MicroObjectives:
    def __init__(self, game, gamma, goal_provider, site_potential, *, retrieval_weight, distance_scale):
        if not 0 < gamma <= 1:
            raise ValueError("Objective rewards require 0 < gamma <= 1")
        self.game, self.gamma = game, gamma
        self.goal_provider, self.site_potential = goal_provider, site_potential
        self.retrieval_weight, self.distance_scale = retrieval_weight, distance_scale
        self.own = {str(c.name): c for c in game.chars if c.team == "A"}
        self.site_best = {}
        self.drop_target, self.drop_distances, self.drop_best = None, None, {}
        self.plant_installments = defaultdict(list)
        self.plant_credited = False
        self.pickups = self.cancellations = self.plants = 0

    def begin_tick(self):
        self.tick_id = int(self.game.battle_tick)
        self.pending = defaultdict(Counter)
        self.shared = Counter()
        self.pickup_names = set()
        self.was_planted = bool(self.game.is_planted)
        self.start_flags = {name: bool(c.has_spike) for name, c in self.own.items()}
        self.start_positions = {name: tuple(c.pos) for name, c in self.own.items() if c.is_alive}
        self.carrier = next((name for name in self.start_positions if self.start_flags[name]), None)
        self.tick_goal = self.goal_provider()
        drop = self.game.spike_pos
        self.tick_drop = tuple(drop) if drop is not None and self.carrier is None and not self.was_planted else None
        if self.tick_drop != self.drop_target:
            self.drop_target = self.tick_drop
            self.drop_distances = distance_map(self.game.grid, self.tick_drop) if self.tick_drop is not None else None
            self.drop_best = {}

    def observe_move(self, char, before):
        if char.team != "A" or before["planted"]:
            return
        name, old_timer = str(char.name), before["timer"]
        completed = (not self.plant_credited and self.game.is_planted and before["has_spike"]
                     and tuple(self.game.planted_pos) == before["pos"])
        if completed:
            advance = max(0, PLANT_REQUIRED_TICKS-old_timer)
        else:
            advance = max(0, min(PLANT_REQUIRED_TICKS, int(char.plant_timer))-old_timer)
        if advance and before["has_spike"]:
            amount = PLANT_PROGRESS_REWARD_PER_TICK*advance
            self.pending[name]["plant_progress"] += amount
            self.plant_installments[name].append((amount, self.tick_id))
        if completed:
            self.plant_credited = True
            self.plants += 1
            self.pending[name]["plant_complete"] += PLANT_COMPLETE_REWARD
            self.plant_installments[name].clear()
        elif old_timer > 0 and int(char.plant_timer) < old_timer:
            self.cancel(name, was_active=True)

    def cancel(self, name, *, was_active=False):
        installments = self.plant_installments[name]
        if not installments and not was_active:
            return
        # Refund in the same discounted units as each earned bonus. Waiting
        # before canceling cannot make the positive half of a cycle profitable.
        refund = sum(amount*self.gamma**(-(self.tick_id-earned)) for amount, earned in installments)
        self.pending[name]["plant_cancel"] -= refund+PLANT_CANCEL_EXTRA_PENALTY
        installments.clear()
        self.cancellations += 1

    def observe_before_shots(self):
        # The engine picks up the spike immediately before resolving shots.
        # Observe here so a pickup followed by death in this tick still counts.
        if self.game.is_planted or self.game.spike_pos is not None:
            return
        for name, char in self.own.items():
            if char.is_alive and char.has_spike and not self.start_flags[name] and name not in self.pickup_names:
                self.pickup_names.add(name)
                self.pending[name]["spike_pickup"] += SPIKE_PICKUP_REWARD
                self.pickups += 1

    def retrieval_proximity(self, positions):
        distances = [int(self.drop_distances[pos]) for pos in positions]
        distances = [value for value in distances if value >= 0]
        return self.retrieval_weight/(1+min(distances)/self.distance_scale) if distances else 0.

    def finish_tick(self, terminal):
        self.observe_before_shots()
        # Score the same actors and fixed target on both sides of movement.
        # A carrier death, pickup, site choice or phase boundary isn't movement.
        goal = self.goal_provider()
        if self.carrier is not None and goal is not None and self.tick_goal in (None, goal):
            before = self.site_potential(self.start_positions[self.carrier], goal)
            after = self.site_potential(tuple(self.own[self.carrier].pos), goal)
            best = max(self.site_best.get(goal, 0.), before)
            self.shared["site_progress"] += max(0., after-best)
            self.site_best[goal] = max(best, after)
        if self.tick_drop is not None:
            for name,position in self.start_positions.items():
                before=self.retrieval_proximity((position,))
                after=self.retrieval_proximity((tuple(self.own[name].pos),))
                best=max(self.drop_best.get(name,0.),before)
                amount=max(0.,after-best)
                if amount:self.pending[name]["retrieval_progress"]+=amount
                self.drop_best[name]=max(best,after)
        if self.pickup_names:
            # A later real drop, even at the same coordinate, starts a new task.
            self.drop_target = None
        if not self.plant_credited:
            for name, installments in self.plant_installments.items():
                char = self.own[name]
                if installments and (terminal or not char.is_alive or int(char.plant_timer) == 0):
                    self.cancel(name)
        return dict(self.shared), {name: dict(parts) for name, parts in self.pending.items()}
