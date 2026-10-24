"""Translate ConCon commands without teaching arbitrary watch-cell choices."""
from collections import Counter
import numpy as np
from ..gc3_engine_actions import MODES, legal_actions, allies, facing_towards
from ..gc3_observation import encode, map_hints
from .gc3_observation_attacker_history import MatchHistory, encode_public, command_features
from .gc3_policy_attacker_history import preplant_legal_actions


class UnrepresentableAction(ValueError):
    pass


def setup_executed_command(char, state, command):
    """Label the action the engine actually executes in setup, not a rejected request."""
    if not state.get("defender_setup_active"):
        return command
    point, payload = command, "MOVE"
    if isinstance(command, tuple) and len(command) == 2 and isinstance(command[1], (str, dict)):
        point, payload = command
    masks, _ = legal_actions(char, state)
    try:
        index = int(np.ravel_multi_index(tuple(map(int, point)), state["grid"].shape))
        allowed = bool(masks["position"][index])
    except (ValueError, TypeError):
        allowed = False
    if not allowed:
        # Rejected movement does not apply the requested facing either.
        return list(char.pos), "MOVE"
    if isinstance(payload, dict) and "facing" in payload:
        return list(point), {"facing":payload["facing"]}
    # Setup executes coordinates only, without abilities/objective channels.
    return list(point), "MOVE"


def teacher_record(char, state, command):
    masks, caps = preplant_legal_actions(char, state)
    hints = map_hints(char, state)
    origin = tuple(map(int, char.pos))
    destination, payload = command, "MOVE"
    if isinstance(command, tuple) and len(command) >= 2 and isinstance(command[1], (str, dict)):
        if len(command) != 2:
            raise UnrepresentableAction("extra_command_payload")
        destination, payload = command
    destination = tuple(map(int, destination))
    if not caps["can_move"]:
        destination = origin
    # A blocked local movement leaves the teacher at its origin. The live
    # engine still applies an explicit facing, so preserve that direction.
    if sum(abs(destination[i]-origin[i]) for i in (0, 1)) <= 1:
        try:
            destination_index = np.ravel_multi_index(destination, state["grid"].shape)
            blocked = not masks["position"][destination_index]
        except ValueError:
            blocked = True
        if blocked:
            destination = origin
    selections = {}
    watch_options = None

    def cell(kind, point):
        try:
            index = int(np.ravel_multi_index(tuple(map(int, point)), state["grid"].shape))
        except (ValueError, TypeError):
            raise UnrepresentableAction("invalid_"+kind) from None
        if not masks[kind][index]:
            raise UnrepresentableAction("illegal_"+kind)
        selections[kind] = index

    def facing(direction, source):
        nonlocal watch_options
        watch_options = np.array([masks["watch"][i] and
            facing_towards(source, np.unravel_index(i, state["grid"].shape), char.facing) == direction
            for i in range(state["grid"].size)], dtype=np.bool_)
        if not watch_options.any():
            raise UnrepresentableAction("unrepresentable_facing")
        selections["watch"] = int(np.flatnonzero(watch_options)[0])

    if isinstance(payload, dict):
        if "ability" in payload:
            mode = "ABILITY"
            if payload["ability"] != char.ability_name:
                raise UnrepresentableAction("wrong_ability")
            if char.ability_name == "DANCE":
                names = [c.name for c in allies(char, state)]
                if payload.get("target_name") not in names:
                    raise UnrepresentableAction("unknown_ally")
                selections["ally"] = names.index(payload["target_name"])
            elif char.ability_name != "RAMP":
                cell("target", payload.get("target"))
        elif "ultimate" in payload:
            mode = "ULTIMATE"
            if payload["ultimate"] != char.ultimate_name:
                raise UnrepresentableAction("wrong_ultimate")
            if char.ultimate_name in ("NEON", "ESCAPE"):
                masks["target"] = masks["ultimate_target"]
                cell("target", payload.get("target"))
            elif char.ultimate_name in ("RAID", "TUNNEL"):
                facing(payload.get("facing", char.facing), origin)
        elif set(payload) == {"facing"}:
            if not caps["can_turn"]:
                # Engine-imposed facing locks ignore this part of the teacher
                # command. Label the equivalent physical move/hold instead.
                mode = "MOVE"
                cell("position", destination)
            else:
                mode = "WATCH" if destination == origin else "MOVE_WATCH"
                if mode == "MOVE_WATCH":
                    cell("position", destination)
                facing(payload["facing"], destination)
        else:
            raise UnrepresentableAction("unknown_payload")
        # Extra parameters can alter engine behavior; do not silently omit them.
        allowed = {"ability", "ultimate", "target", "target_name"}
        if set(payload) == {"facing"} or (mode == "ULTIMATE" and char.ultimate_name in ("RAID", "TUNNEL")):
            allowed.add("facing")
        if set(payload)-allowed:
            raise UnrepresentableAction("unsupported_payload_keys")
    else:
        mode = "MOVE" if payload in ("MOVE", "HOLD") else payload
        if mode == "MOVE":
            cell("position", origin if payload == "HOLD" else destination)
    if mode not in MODES or not masks["mode"][MODES.index(mode)]:
        raise UnrepresentableAction("illegal_mode_"+str(mode))
    selections["mode"] = MODES.index(mode)
    for kind, action in selections.items():
        if not masks[kind][action]:
            raise UnrepresentableAction("illegal_"+kind)
    return dict(obs=encode(char, state, hints), hints=hints, masks=masks, phase="standard",
                selections=selections, watch_options=watch_options,
                carrying=bool(char.has_spike), retrieving=state.get("spike_pos") is not None)


class RecordingTeacher:
    requires_public_effects = True

    def __init__(self, teacher, history=None, executing_policy=None):
        self.teacher, self.rows, self.skipped = teacher, [], Counter()
        self.history = history if history is not None else MatchHistory()
        self.executing_policy = executing_policy

    def set_game(self, game):
        self.teacher.set_game(game)

    def reset_round(self):
        self.rows.clear()
        self.skipped.clear()
        self.teacher.reset_round()
        self.history.begin_round()

    def on_round_finished(self, result):
        self.history.finish_round(result)

    def decide_move(self, char, state):
        if state.get("is_planted"):
            return list(char.pos), "HOLD"
        observation = encode_public(char, state, self.history)
        goal = getattr(self.teacher, "target_site", None)
        first_site_choice = goal is not None and self.history.round_id not in self.history.goal_sites
        if goal is not None and self.executing_policy is None:
            self.history.goal_sites[self.history.round_id] = int(goal)
        command = setup_executed_command(char, state, self.teacher.decide_move(char, state))
        executed = self.executing_policy.select(char, state, self.history)[0] if self.executing_policy is not None else command
        if char.is_alive:
            index = len(self.history.events)
            try:
                row = teacher_record(char, state, command)
            except UnrepresentableAction as error:
                self.skipped[str(error)] += 1
            else:
                row.update(obs=observation, history=self.history, history_index=index,
                           agent=str(char.name), tick=int(state.get("battle_tick", 0)))
                if goal is not None:
                    row["goal_site"] = int(goal)
                    if first_site_choice:
                        row["selections"]["site"] = int(goal)
                        row["masks"]["site"] = np.ones(2, dtype=np.bool_)
                self.rows.append(row)
            # DAgger labels the teacher action, but remembers the student's actual command.
            self.history.append(observation, command_features(char, executed, state), name=char.name,
                                tick=state.get("battle_tick", 0), command=executed)
        return executed
