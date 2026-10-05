"""Perception-based search network and legal movement/utility candidates.

The frozen foundation owns quiet/setup navigation. Tactical actions are model
predictions, including stationary fighting, returning and assisting. No BFS
route or behavioral override is used here.
"""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from game_core import FACING_DIRECTIONS, SHOOT_INTERVAL_TICKS
from concon_v1.co1_defender_common import (
    DefenderSearchDQN, build_inputs, observation_dim as basic_observation_dim,
    FEATURE_DIM as BASIC_FEATURE_DIM, MOVES, GORIGONS,
)
from concon_v1.co1_guard_common import clear_shot, aim_alignment, facing_onehot
from concon_v1.co1_attacker_abilities import _impact_aim

MEMORY_TICKS = 12
POST_KILL_HOLD_TICKS = 6
SUPPORT_DISTANCE = 6
TARGET_COUNT = 5
ABILITIES = ("SMOKE", "FLASH", "RECON")
ACTION_DIM = 40 + len(ABILITIES) * TARGET_COUNT
SELF_FEATURES = 18
ENEMY_FEATURES = 21
LOCAL_FEATURES = 3
TACTICAL_FEATURE_DIM = SELF_FEATURES + TARGET_COUNT * ENEMY_FEATURES + 5 * LOCAL_FEATURES
POLICY_TYPE = "concon_defender_search_v1"


def observation_dim(scenario):
    return basic_observation_dim(scenario) + 2 * scenario.grid.size + TACTICAL_FEATURE_DIM


class DefenderSearchBattleDQN(DefenderSearchDQN):
    combat = True

    def __init__(self, scenario):
        super().__init__(scenario)
        self.basic_dim = basic_observation_dim(scenario)
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 * 4 * 6 + BASIC_FEATURE_DIM + TACTICAL_FEATURE_DIM, 256),
                                  nn.ReLU(), nn.Linear(256, ACTION_DIM))
        self.freeze_positioning()

    def freeze_positioning(self):
        self.navigation_values.weight.requires_grad_(False)
        self.navigation_yield_values.weight.requires_grad_(False)

    def forward(self, observation):
        basic = observation[:, :self.basic_dim]
        map_end = self.basic_dim + 2 * self.height * self.width
        tactical = observation[:, map_end:]
        maps = torch.cat((basic[:, :self.map_size].reshape(-1, 6, self.height, self.width),
                          observation[:, self.basic_dim:map_end].reshape(-1, 2, self.height, self.width)), dim=1)
        values = self.head(torch.cat((self.encoder(maps), basic[:, self.map_size:], tactical), dim=1))
        navigation = super().forward(basic)
        quiet_values = torch.cat((navigation, navigation.new_full((len(navigation), ACTION_DIM - 40), -1e6)), dim=1)
        quiet = (tactical[:, 0] == 0) | (basic[:, self.map_size + 2] > 0)
        if getattr(self, "foundation_only", False):
            quiet = torch.ones_like(quiet)
        return torch.where(quiet[:, None], quiet_values, values)


def load_search_weights(model, checkpoint, scenario):
    from concon_v1.co1_defender_scenario import validate_checkpoint
    validate_checkpoint(checkpoint, scenario)
    if tuple(checkpoint.get("training_roster", ())) != GORIGONS.players:
        raise ValueError("defender search checkpoint roster mismatch")
    basic = checkpoint["policy_type"] == "concon_defender_search_positioning_v1"
    expected_obs = basic_observation_dim(scenario) if basic else observation_dim(scenario)
    expected_actions = 40 if basic else ACTION_DIM
    if checkpoint.get("obs_dim") != expected_obs or checkpoint.get("n_actions") != expected_actions:
        raise ValueError("defender search checkpoint dimensions mismatch")
    if basic:
        # Exact old positioning weights, with only the new network initialized.
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        expected_missing = {key for key in model.state_dict() if key.startswith(("encoder.", "head."))}
        if set(missing) != expected_missing or unexpected:
            raise ValueError("incompatible defender foundation weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.freeze_positioning()


def _located(other, height, width):
    return (getattr(other, "position_known", False) and other.is_alive
            and 0 <= other.pos[0] < height and 0 <= other.pos[1] < width)


def _at(char, position):
    """Detached geometry probe; never move or mutate a real/perceived actor."""
    return SimpleNamespace(name=char.name, pos=list(position), team=char.team,
                           facing=char.facing, is_alive=True,
                           sees_through_smoke=getattr(char, "sees_through_smoke", False),
                           reveal_remaining=getattr(char, "reveal_remaining", 0))


def build_search_inputs(controller, char, state):
    basic, movement_mask = build_inputs(controller, char, state)
    scenario, grid = controller.scenario, np.asarray(state["grid"])
    height, width = grid.shape
    tick = int(state.get("battle_tick", 0))
    setup = bool(state.get("defender_setup_active", False))
    position = tuple(map(int, char.pos))
    chars = state.get("chars", [])
    allies = [other for other in chars if other.team == char.team and other.is_alive]
    disclosed = sorted([other for other in chars if other.team != char.team and _located(other, height, width)],
                       key=lambda other: other.name)
    smoke = set(map(tuple, state.get("smoke_cells", ())))
    game = getattr(controller, "game", None)
    if game is not None:
        smoke.update(game._smoke_cells())
    if not setup:
        for enemy in disclosed:
            controller.sightings[enemy.name] = dict(pos=tuple(enemy.pos), tick=tick, facing=enemy.facing,
                hp=float(getattr(enemy, "hp", 100)) / max(1, getattr(enemy, "max_hp", 100)),
                blind=float(getattr(enemy, "blind_remaining", 0)),
                reveal=float(getattr(enemy, "reveal_remaining", 0)), alive=True)
    for name, memory in list(controller.sightings.items()):
        if tick < memory["tick"] or tick - memory["tick"] >= MEMORY_TICKS:
            del controller.sightings[name]
    # IQ omission can set is_alive=False on a known, living enemy. Only an
    # unknown/dead proxy is an unambiguous public death report here. Preserve
    # the last disclosed position; never read a corpse's hidden position.
    for enemy in chars:
        if (enemy.team != char.team and not enemy.is_alive
                and not getattr(enemy, "position_known", False)
                and enemy.name in controller.sightings):
            controller.sightings[enemy.name]["alive"] = False
    fireable = [enemy for enemy in disclosed if clear_shot(char, enemy, chars, grid, smoke)]
    fireable.sort(key=lambda enemy: (np.linalg.norm(np.subtract(enemy.pos, position)), enemy.name))
    engaged = [enemy for enemy in disclosed if any(
        ally.name != char.name and clear_shot(ally, enemy, chars, grid, smoke)
        and aim_alignment(ally.pos, enemy.pos, ally.facing) >= .7
        and getattr(ally, "blind_remaining", 0) <= 0 for ally in allies)]
    target = fireable[0] if fireable else None
    remembered = sorted(((name, memory) for name, memory in controller.sightings.items()
                         if memory["alive"]), key=lambda item: (
        -item[1]["tick"], np.linalg.norm(np.subtract(item[1]["pos"], position)), item[0]))
    aim_target = tuple(target.pos) if target is not None else remembered[0][1]["pos"] if remembered else None
    slot = controller.assignments[char.name]
    goal, aim = scenario.positions[slot], scenario.facing_points[slot]
    stopped = controller.stationary_ticks(char, tick)
    previous = controller.combat_history.get(char.name)
    after_shot = bool(previous and previous["tick"] < tick and previous["fireable"]
                      and tick % SHOOT_INTERVAL_TICKS == 0 and not getattr(char, "moved_this_tick", False))
    killed = bool(previous and getattr(char, "round_kills", 0) > previous["kills"])
    shot_target_name = previous.get("target_name") if previous else None
    if game is not None and hasattr(game, "last_shots"):
        # Read only this actor's public firing result. In particular never use
        # the raw shot target's position/HP or any opponent's shot records.
        own_shots = [shot for shot in game.last_shots
                     if shot["shooter"].name == char.name and shot["shooter"].team == char.team]
        after_shot = bool(own_shots) and tick % SHOOT_INTERVAL_TICKS == 0
        shot_target_name = own_shots[-1]["target"].name if own_shots else None
    if killed and shot_target_name in controller.sightings:
        controller.sightings[shot_target_name]["alive"] = False
        remembered = [(name, memory) for name, memory in remembered if memory["alive"]]
        if target is None:
            aim_target = remembered[0][1]["pos"] if remembered else None
    watch_position = previous.get("watch_position") if previous else None
    watch_until = previous.get("watch_until", -1) if previous else -1
    killed_position = (controller.sightings[shot_target_name]["pos"]
                       if killed and shot_target_name in controller.sightings
                       else previous.get("target_position") if previous else None)
    if killed and killed_position is not None:
        watch_position = killed_position
        watch_until = tick + POST_KILL_HOLD_TICKS
    post_kill_hold = bool(not setup and target is None and watch_position is not None and tick < watch_until)
    if post_kill_hold:
        watch_probe = SimpleNamespace(name="__search_memory__", pos=watch_position, reveal_remaining=0)
        post_kill_hold = clear_shot(_at(char, position), watch_probe, chars, grid, smoke)
    if post_kill_hold:
        aim_target = watch_position
    controller.combat_history[char.name] = dict(tick=tick, fireable=target is not None,
        target_name=target.name if target is not None else None,
        target_position=tuple(target.pos) if target is not None else None,
        watch_position=watch_position, watch_until=watch_until, kills=getattr(char, "round_kills", 0))
    active = (bool(remembered) or post_kill_hold) and not setup
    extras = np.zeros((2, height, width), dtype=np.float32)
    for memory in controller.sightings.values():
        extras[0, memory["pos"][0], memory["pos"][1]] = 1 - (tick - memory["tick"]) / MEMORY_TICKS
    for r, c in smoke:
        if 0 <= r < height and 0 <= c < width:
            extras[1, r, c] = 1
    # The existing kill indicator stays set throughout the observable watch
    # interval. Previously the policy lost this signal after just one tick.
    features = [float(active), min(stopped, 4) / 4, float(after_shot), float(post_kill_hold),
                float(getattr(char, "hp", 100)) / max(1, getattr(char, "max_hp", 100)),
                min(float(getattr(char, "blind_remaining", 0)), 15) / 15,
                min(getattr(char, "smoke_charges", 0), 2) / 2,
                min(getattr(char, "flash_charges", 0), 2) / 2,
                min(getattr(char, "recon_charges", 0), 2) / 2,
                goal[0] / height, goal[1] / width, aim[0] / height, aim[1] / width,
                aim_target[0] / height if aim_target else 0, aim_target[1] / width if aim_target else 0,
                float(target is not None), float(bool(engaged)),
                ((tick + 1) % SHOOT_INTERVAL_TICKS) / max(1, SHOOT_INTERVAL_TICKS)]
    for name, memory in sorted(controller.sightings.items())[:TARGET_COUNT]:
        enemy = next((enemy for enemy in disclosed if enemy.name == name), None)
        age = tick - memory["tick"]
        features += [1., float(enemy is not None), float(memory["alive"]),
                     memory["pos"][0] / height, memory["pos"][1] / width]
        features += facing_onehot(memory["facing"])
        features += [memory["hp"], max(0., memory["blind"] - age) / 15,
                     max(0., memory["reveal"] - age) / 15, age / MEMORY_TICKS,
                     float(enemy in fireable) if enemy is not None else 0.,
                     float(enemy in engaged) if enemy is not None else 0.,
                     float(enemy is not None and aim_alignment(enemy.pos, position, enemy.facing) >= .7),
                     float(memory["pos"][1] >= width / 2)]
    features += [0.] * (SELF_FEATURES + TARGET_COUNT * ENEMY_FEATURES - len(features))
    threats, lines = [], []
    for move, (dr, dc) in enumerate(MOVES):
        destination = (position[0] + dr, position[1] + dc)
        probe = _at(char, destination)
        legal = movement_mask[move * 8:(move + 1) * 8].any()
        line = legal and any(clear_shot(probe, enemy, chars, grid, smoke) for enemy in disclosed)
        threat = legal and any(clear_shot(enemy, probe, chars, grid, smoke)
                              and aim_alignment(enemy.pos, destination, enemy.facing) >= .7
                              and getattr(enemy, "blind_remaining", 0) <= 0 for enemy in disclosed)
        features += [float(legal), float(line), float(threat)]
        lines.append(bool(line))
        threats.append(bool(threat))
    if len(features) != TACTICAL_FEATURE_DIM:
        raise RuntimeError("defender tactical feature dimensions changed")
    observation = np.concatenate((basic, extras.ravel(), np.asarray(features, dtype=np.float32)))
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[:40] = movement_mask
    payloads = {}
    # Candidate geometry matches attacker FLASH targeting; availability never
    # forces a cast. Smoke remains selectable so conserving it is learned.
    for index, enemy in enumerate(disclosed[:TARGET_COUNT]):
        for ability_index, ability in enumerate(ABILITIES):
            if (setup or getattr(char, "ability_name", "") != ability
                    or getattr(char, ability.lower() + "_charges", 0) <= 0):
                continue
            if ability == "FLASH":
                if enemy not in engaged or game is None:
                    continue
                destination = _impact_aim(game, char, tuple(enemy.pos), radius=max(grid.shape), require_flash_hit=enemy)
            elif ability == "RECON":
                destination = _impact_aim(game, char, tuple(enemy.pos), radius=4) if game is not None else None
            else:
                destination = tuple(enemy.pos)
            if destination is None:
                continue
            r, c = destination
            if not (0 <= r < height and 0 <= c < width) or grid[r, c] == 1:
                continue
            action = 40 + ability_index * TARGET_COUNT + index
            mask[action] = True
            payloads[action] = dict(ability=ability, target=destination)
    context = dict(position=position, goal=goal, aim=aim, target=aim_target,
                   target_name=target.name if target is not None else None,
                   fireable=target is not None, neutralized=bool(target is not None and getattr(target, "blind_remaining", 0) > 0),
                   after_shot=after_shot, killed=killed, stopped=stopped, tick=tick, setup=setup,
                   active=active, post_kill_hold=post_kill_hold, threats=threats, lines=lines, retreat_available=threats[4] and any(
                       movement_mask[index * 8:(index + 1) * 8].any() and not threats[index] for index in range(4)),
                   chars=chars, actor=_at(char, position), disclosed=disclosed, engaged=engaged,
                   smoke=smoke, ability_payloads=payloads)
    return observation, mask, context


def decode_action(action, char, context):
    if action >= 40:
        # Ability ticks keep the actor's facing in the actual engine.
        return list(char.pos), {**context["ability_payloads"][action], "facing": char.facing}
    move, facing = divmod(action, 8)
    dr, dc = MOVES[move]
    return [char.pos[0] + dr, char.pos[1] + dc], {"facing": FACING_DIRECTIONS[facing]}
