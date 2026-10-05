"""Production inference for the defender's learned basic positioning skill."""

import random
from pathlib import Path

import torch

from controllers import BaseController
from game_core import FACING_DIRECTIONS
from concon_v1.co1_defender_scenario import get_scenario, validate_checkpoint
from concon_v1.co1_defender_common import DefenderSearchDQN, build_inputs, observation_dim, ACTION_DIM, MOVES, GORIGONS


class ConconDefenderSearchController(BaseController):
    def __init__(self, model_path=None, model=None, seed=0):
        super().__init__()
        self.scenario = get_scenario()
        self.rng = random.Random(seed)
        self.model_path = None
        if model is None:
            path = Path(model_path) if model_path is not None else self.scenario.runtime_model_path
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, self.scenario)
            combat = checkpoint.get("policy_type") == "concon_defender_search_v1"
            if combat:
                from concon_v1.co1_defender_search_common import DefenderSearchBattleDQN, load_search_weights
                model = DefenderSearchBattleDQN(self.scenario)
                load_search_weights(model, checkpoint, self.scenario)
            elif (checkpoint.get("obs_dim") != observation_dim(self.scenario)
                    or checkpoint.get("n_actions") != ACTION_DIM
                    or tuple(checkpoint.get("training_roster", ())) != GORIGONS.players):
                raise ValueError("defender checkpoint dimensions/roster mismatch")
            else:
                model = DefenderSearchDQN(self.scenario)
                model.load_state_dict(checkpoint["model_state_dict"])
            self.model_path = path.resolve()
            print(f"[ConCon defender search] model={self.model_path} "
                  f"phase=setup/search skill={'search' if combat else 'positioning'}", flush=True)
        self.model = model
        self.model.eval()
        self.reset_round()

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.assignments = {}
        self.sightings = {}
        self.combat_history = {}
        self._stationary = {}
        self.last_context = {}

    def stationary_ticks(self, char, tick):
        position = tuple(char.pos)
        previous = self._stationary.get(char.name)
        if previous is not None and previous[0] == tick:
            return previous[2]
        count = (previous[2] + 1 if previous is not None and previous[0] == tick - 1
                 and previous[1] == position and not getattr(char, "moved_this_tick", False) else 0)
        self._stationary[char.name] = (tick, position, count)
        return count

    def prepare_assignments(self, chars):
        if self.assignments:
            return
        names = [other.name for other in chars if other.team == "D" and other.is_alive]
        if not names or len(names) > 5 or any(name not in GORIGONS.players for name in names):
            raise ValueError("defender search requires Gorigons defenders")
        slots = list("abcde")
        self.rng.shuffle(slots)
        self.assignments = dict(zip(names, slots))

    def policy_inputs(self, char, state):
        self.prepare_assignments(state.get("chars", []))
        if getattr(self.model, "combat", False):
            from concon_v1.co1_defender_search_common import build_search_inputs
            observation, mask, self.last_context = build_search_inputs(self, char, state)
            return observation, mask
        self.last_context = {"active": False, "setup": bool(state.get("defender_setup_active"))}
        return build_inputs(self, char, state)

    def choose_action(self, char, observation, mask, context):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            values = self.model(torch.as_tensor(observation, device=device).unsqueeze(0))[0]
            return int(values.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf).argmax())

    def decide_move(self, char, game_state):
        observation, mask = self.policy_inputs(char, game_state)
        action = self.choose_action(char, observation, mask, self.last_context)
        if getattr(self.model, "combat", False):
            from concon_v1.co1_defender_search_common import decode_action
            return decode_action(action, char, self.last_context)
        move, facing = divmod(action, 8)
        dr, dc = MOVES[move]
        return [char.pos[0] + dr, char.pos[1] + dc], {"facing": FACING_DIRECTIONS[facing]}
