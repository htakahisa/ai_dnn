"""
Ghost Champions v1 - Attacker Macro runtime controller.

This is the practical-game adapter for the Macro DQN trained by
train_attacker_macro_gc_v28.py.

Design:
- Macro DQN chooses team strategy every 3 battle ticks.
- MacroEnv from the training file is reused for the exact 68-dim observation,
  action mask, strategy assignments, Split/Fake/Rotate option state, and v28
  Rotate Opportunity logic.
- Real game positions/alive/spike state are synchronized into the shadow env.
- Hidden defenders are NOT copied into the Macro model. Enemy information is
  updated only from defenders that are revealed or have line of sight to at
  least one living attacker.
- Carry/Escort controllers still get first chance to emit PLANT/ABILITY.
  Normal MOVE is overridden by the team macro target.
- Post-plant Guard and dropped-spike Retrieve remain outside this controller.
"""

from __future__ import annotations

from pathlib import Path
from collections import deque
import os

import numpy as np
import torch

# Keep this import exact: runtime must use the same observation/action semantics
# as the final Macro model.
from train_attacker_macro_gc_v28 import (
    MacroEnv,
    MacroDuelingDQN,
    OBS_DIM,
    N_ACTIONS,
    STRATEGIES,
    STRATEGY_TO_INDEX,
    SIDE_A,
    SIDE_B,
    SIDE_MID,
    LOW_LEVEL_TICKS_PER_MACRO_STEP,
    ROUND_DURATION_TICKS,
    INFO_DECAY,
    INFO_CONF_LOW,
    INFO_CONF_MEDIUM,
    INFO_CONF_HIGH,
    INFO_GAIN_FORWARD_CONTROL,
    INFO_GAIN_INFO_AREA_V21,
    INFO_GAIN_DEEP_CONTROL,
    INFO_GAIN_SITE_V21,
    INFO_EST_ALPHA_LOW,
    INFO_EST_ALPHA_MEDIUM,
    INFO_EST_ALPHA_HIGH,
    CONTROL_GAIN,
    CONTROL_DECAY,
    _forward_control_cells,
    _info_cells,
    _deep_control_cells,
    _site_cells,
    side_of_pos,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = (
    HERE / "data" / "attacker_macro_gc_data" / "dqn_attacker_macro_gc_final.pt",
    HERE / "data" / "attacker_macro_gc_data" / "dqn_attacker_macro_gc_best_by_eval.pt",
    HERE / "data" / "attacker_macro_gc_data" / "dqn_attacker_macro_gc_latest.pt",
)

CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.is_file():
            return p
    return None


def _line_cells(p1, p2):
    y0, x0 = int(p1[0]), int(p1[1])
    y1, x1 = int(p2[0]), int(p2[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    cells = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _has_los(grid, p1, p2):
    for r, c in _line_cells(p1, p2):
        if int(grid[r, c]) == 1:
            return False
    return True


def _bfs_next_step(grid, start, goal, occupied):
    """One legal step toward goal, avoiding current occupied cells."""
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    if start == goal:
        return start

    h, w = grid.shape
    q = deque([start])
    parent = {start: None}
    reached = None

    while q:
        cur = q.popleft()
        if cur == goal:
            reached = cur
            break
        r, c = cur
        for dr, dc in CARDINAL:
            nxt = (r + dr, c + dc)
            if nxt in parent:
                continue
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if int(grid[nxt[0], nxt[1]]) == 1:
                continue
            if nxt in occupied and nxt != goal:
                continue
            parent[nxt] = cur
            q.append(nxt)

    if reached is None:
        # If exact goal is occupied/unreachable, use a neighbor that lowers
        # Manhattan distance and is legal.
        r, c = start
        candidates = []
        for dr, dc in CARDINAL:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if int(grid[nxt[0], nxt[1]]) == 1 or nxt in occupied:
                continue
            d = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
            candidates.append((d, nxt))
        return min(candidates)[1] if candidates else start

    step = reached
    while parent[step] is not None and parent[step] != start:
        step = parent[step]
    if parent[step] is None:
        return start
    return step


class LearningAttackerMacroGCController:
    """Team-level Macro coordinator used on top of GC Carry/Escort."""

    def __init__(
        self,
        model_path=None,
        device="auto",
        greedy=True,
        verbose=True,
    ):
        self.greedy = bool(greedy)
        self.verbose = bool(verbose)
        self.game = None

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        if model_path is None:
            model_path = _first_existing(DEFAULT_MODEL_CANDIDATES)
        else:
            model_path = Path(model_path)

        if model_path is None or not Path(model_path).is_file():
            raise FileNotFoundError(
                "Macro model not found. Expected one of: "
                + ", ".join(str(x) for x in DEFAULT_MODEL_CANDIDATES)
            )

        self.model_path = Path(model_path)
        checkpoint = torch.load(
            str(self.model_path),
            map_location=self.device,
            weights_only=False,
        )

        obs_dim = int(checkpoint.get("obs_dim", OBS_DIM))
        n_actions = int(checkpoint.get("n_actions", N_ACTIONS))
        if obs_dim != OBS_DIM or n_actions != N_ACTIONS:
            raise ValueError(
                f"Macro checkpoint mismatch: obs_dim={obs_dim}/{OBS_DIM}, "
                f"n_actions={n_actions}/{N_ACTIONS}"
            )

        saved_strategies = checkpoint.get("strategies")
        if saved_strategies is not None and list(saved_strategies) != list(STRATEGIES):
            raise ValueError(
                "Macro checkpoint strategy order does not match runtime v28."
            )

        self.model = MacroDuelingDQN(obs_dim, n_actions).to(self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

        self.env = MacroEnv()
        self._last_real_tick = None
        self._last_macro_decision_tick = None
        self._last_strategy = None
        self._last_q_values = None

        if self.verbose:
            print(
                "[GC Macro Runtime] loaded "
                f"{self.model_path.name} "
                f"(episode={checkpoint.get('episode')}, "
                f"obs={obs_dim}, actions={n_actions})"
            )

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.env.reset(
            forced_strategy="DEFAULT",
            forced_curriculum_mode="FREE",
        )
        self._last_real_tick = None
        self._last_macro_decision_tick = None
        self._last_strategy = self.env.current_strategy
        self._last_q_values = None

    # ------------------------------------------------------------------
    # Real-state synchronization
    # ------------------------------------------------------------------

    def _real_attackers(self, game_state):
        return [
            c for c in game_state.get("chars", [])
            if getattr(c, "team", None) == "A"
        ]

    def _real_defenders(self, game_state):
        return [
            c for c in game_state.get("chars", [])
            if getattr(c, "team", None) == "D"
        ]

    def _sync_attackers(self, game_state):
        real = {c.name: c for c in self._real_attackers(game_state)}
        shadow_by_name = {a.name: a for a in self.env.attackers}

        # GC roster normally matches names exactly. Fallback by roster order.
        real_list = self._real_attackers(game_state)
        for i, shadow in enumerate(self.env.attackers):
            src = real.get(shadow.name)
            if src is None and i < len(real_list):
                src = real_list[i]
            if src is None:
                shadow.is_alive = False
                continue
            shadow.pos = tuple(map(int, src.pos))
            shadow.is_alive = bool(getattr(src, "is_alive", True))
            shadow.has_spike = bool(getattr(src, "has_spike", False))

    def _visible_defender_counts(self, game_state):
        grid = np.asarray(game_state["grid"])
        attackers = [
            c for c in self._real_attackers(game_state)
            if bool(getattr(c, "is_alive", True))
        ]
        counts = {SIDE_A: 0, SIDE_B: 0, SIDE_MID: 0}

        for d in self._real_defenders(game_state):
            if not bool(getattr(d, "is_alive", True)):
                continue

            revealed = bool(
                getattr(d, "revealed", False)
                or getattr(d, "is_revealed", False)
                or getattr(d, "reveal_timer", 0)
                or getattr(d, "revealed_ticks", 0)
            )
            visible = revealed or any(
                _has_los(grid, a.pos, d.pos)
                for a in attackers
            )
            if visible:
                counts[side_of_pos(tuple(map(int, d.pos)))] += 1

        return counts

    def _runtime_area_coverage(self, cells):
        return self.env._area_coverage(cells)

    def _update_information_from_real_game(self, game_state):
        """Update v28 info/control without copying hidden defender positions."""
        visible_counts = self._visible_defender_counts(game_state)

        for side in (SIDE_A, SIDE_B, SIDE_MID):
            self.env._info_memory_age[side] += 1
            self.env.info_conf[side] = max(
                0.0,
                float(self.env.info_conf[side]) - INFO_DECAY,
            )

            forward_cov = self._runtime_area_coverage(
                _forward_control_cells(side)
            )
            info_cov = self._runtime_area_coverage(_info_cells(side))
            deep_cov = self._runtime_area_coverage(
                _deep_control_cells(side)
            )
            site_cov = (
                self._runtime_area_coverage(_site_cells(side))
                if side in {SIDE_A, SIDE_B}
                else 0.0
            )

            gain = (
                forward_cov * INFO_GAIN_FORWARD_CONTROL
                + info_cov * INFO_GAIN_INFO_AREA_V21
                + deep_cov * INFO_GAIN_DEEP_CONTROL
                + site_cov * INFO_GAIN_SITE_V21
            )

            floor = 0.0
            if forward_cov > 0.0:
                floor = max(floor, INFO_CONF_LOW)
            if info_cov > 0.0:
                floor = max(floor, INFO_CONF_HIGH)
            if deep_cov > 0.0:
                floor = max(floor, 0.74)
            if site_cov > 0.0:
                floor = max(floor, 0.82)

            if gain > 0.0 or floor > 0.0:
                self.env.info_conf[side] = min(
                    1.0,
                    max(
                        floor,
                        float(self.env.info_conf[side]) + gain,
                    ),
                )

                conf = float(self.env.info_conf[side])
                if conf >= INFO_CONF_HIGH:
                    alpha = INFO_EST_ALPHA_HIGH
                elif conf >= INFO_CONF_MEDIUM:
                    alpha = INFO_EST_ALPHA_MEDIUM
                else:
                    alpha = INFO_EST_ALPHA_LOW

                observed_count = float(visible_counts[side])
                self.env.enemy_est[side] = (
                    float(self.env.enemy_est[side]) * (1.0 - alpha)
                    + observed_count * alpha
                )

                self.env._info_memory_count[side] = float(
                    self.env.enemy_est[side]
                )
                self.env._info_memory_conf[side] = conf
                self.env._info_memory_age[side] = 0

            for depth_name, cells in (
                ("FORWARD", _forward_control_cells(side)),
                ("DEEP", _deep_control_cells(side)),
            ):
                key = f"{side}_{depth_name}"
                self.env.control[key] = max(
                    0.0,
                    float(self.env.control[key]) - CONTROL_DECAY,
                )
                cov = self._runtime_area_coverage(cells)
                self.env.control[key] = min(
                    1.0,
                    float(self.env.control[key]) + CONTROL_GAIN * cov,
                )

            self.env.pressure[side] = min(
                1.0,
                0.35 * forward_cov
                + 0.45 * deep_cov
                + 0.70 * site_cov,
            )

        self.env.map_control_score = float(
            np.mean(
                [
                    self.env.control[f"{SIDE_A}_FORWARD"],
                    self.env.control[f"{SIDE_B}_FORWARD"],
                    self.env.control[f"{SIDE_MID}_FORWARD"],
                    self.env.control[f"{SIDE_MID}_DEEP"],
                ]
            )
        )

    # ------------------------------------------------------------------
    # Macro decision / option progression
    # ------------------------------------------------------------------

    def _tick_id(self):
        if self.game is not None:
            return int(getattr(self.game, "battle_tick", 0))
        return 0

    def _sync_tick_once(self, game_state):
        tick_id = self._tick_id()
        if self._last_real_tick == tick_id:
            return

        self._last_real_tick = tick_id
        self._sync_attackers(game_state)

        self.env.tick = min(
            int(tick_id),
            int(ROUND_DURATION_TICKS),
        )
        self.env.macro_step = max(
            0,
            int(tick_id // max(1, LOW_LEVEL_TICKS_PER_MACRO_STEP)),
        )

        self._update_information_from_real_game(game_state)

        # Advance internal option phases against REAL positions, not simulated
        # MacroEnv movement/combat.
        for a in self.env._living_attackers():
            if hasattr(self.env, "_advance_assignment_phase_if_needed"):
                self.env._advance_assignment_phase_if_needed(a)

        if hasattr(self.env, "_update_fake_option_phase"):
            self.env._update_fake_option_phase()
        if hasattr(self.env, "_update_tactical_history"):
            self.env._update_tactical_history()

        self._maybe_decide_macro(tick_id)

    def _maybe_decide_macro(self, tick_id):
        macro_interval = max(1, LOW_LEVEL_TICKS_PER_MACRO_STEP)
        if (
            self._last_macro_decision_tick is not None
            and tick_id - self._last_macro_decision_tick < macro_interval
        ):
            return

        self._last_macro_decision_tick = int(tick_id)
        obs = self.env.build_observation()
        mask = self.env.action_mask()

        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(x).squeeze(0).cpu().numpy()

        q_masked = q.copy()
        q_masked[~mask] = -1e9
        action_idx = int(np.argmax(q_masked))
        strategy = STRATEGIES[action_idx]
        self._last_q_values = q.copy()

        if strategy != self.env.current_strategy:
            old_strategy = self.env.current_strategy

            if old_strategy in {"ROTATE_A_TO_B", "ROTATE_B_TO_A"}:
                if hasattr(self.env, "_clear_rotate_intent_v26"):
                    self.env._clear_rotate_intent_v26(
                        expired=not bool(
                            getattr(self.env, "_smart_rotate_completed", False)
                        )
                    )

            self.env.previous_strategy = old_strategy
            self.env.current_strategy = strategy
            self.env.strategy_age = 0
            self.env._default_info_option_completion_counted = False
            self.env._apply_strategy_assignments(
                strategy,
                initial=False,
            )

            if strategy in {"ROTATE_A_TO_B", "ROTATE_B_TO_A"}:
                if hasattr(self.env, "_clear_rotate_intent_v26"):
                    self.env._clear_rotate_intent_v26(expired=False)
                if hasattr(self.env, "_create_rotate_intent_v26"):
                    self.env._create_rotate_intent_v26(strategy)

            self._last_strategy = strategy
            self._retarget_real_plant_position(strategy)

            if self.verbose:
                print(
                    "[GC Macro] "
                    f"tick={tick_id} {old_strategy} -> {strategy}"
                )
        else:
            self.env.strategy_age += 1

    # ------------------------------------------------------------------
    # Plant target adaptation
    # ------------------------------------------------------------------

    @staticmethod
    def _strategy_target_side(strategy):
        if strategy in {
            "A_RUSH",
            "A_SPLIT",
            "FAKE_B_TO_A",
            "ROTATE_B_TO_A",
            "REHIT_A",
        }:
            return SIDE_A
        if strategy in {
            "B_RUSH",
            "MID_TO_B",
            "B_SPLIT",
            "FAKE_A_TO_B",
            "ROTATE_A_TO_B",
            "REHIT_B",
        }:
            return SIDE_B
        return None

    def _retarget_real_plant_position(self, strategy):
        """Allow actual A/B rotations by moving the round's single plant target.

        Existing Carry model still sees one fixed target at a time. We only
        change it when the Macro strategy changes target site.
        """
        if self.game is None:
            return

        side = self._strategy_target_side(strategy)
        if side is None:
            side = getattr(self.env, "target_site", None)
        if side not in {SIDE_A, SIDE_B}:
            return

        grid = np.asarray(self.game.grid)
        plant_cells = [
            tuple(map(int, p))
            for p in zip(*np.where(grid == 2))
            if side_of_pos(tuple(map(int, p))) == side
        ]
        if not plant_cells:
            return

        carrier = next(
            (
                c for c in getattr(self.game, "chars", [])
                if getattr(c, "team", None) == "A"
                and bool(getattr(c, "is_alive", True))
                and bool(getattr(c, "has_spike", False))
            ),
            None,
        )
        origin = tuple(carrier.pos) if carrier is not None else plant_cells[0]
        chosen = min(
            plant_cells,
            key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]),
        )

        self.game.target_plant_pos = tuple(chosen)

    # ------------------------------------------------------------------
    # Public coordinate API
    # ------------------------------------------------------------------

    @staticmethod
    def _is_special_phase_result(result):
        """True for PLANT/ABILITY-like outputs that Macro must not override."""
        if not isinstance(result, tuple) or len(result) < 2:
            return False

        second = result[1]
        if isinstance(second, dict) and second.get("ability"):
            return True
        if isinstance(second, str) and second.upper() in {
            "PLANT",
            "ABILITY",
            "DEFUSE",
        }:
            return True
        return False

    def coordinate(self, char, game_state, base_result):
        """Return base PLANT/ABILITY, otherwise move one step to Macro target."""
        if bool(game_state.get("is_planted", False)):
            return base_result

        holder = next(
            (
                c for c in game_state.get("chars", [])
                if getattr(c, "team", None) == "A"
                and bool(getattr(c, "is_alive", True))
                and bool(getattr(c, "has_spike", False))
            ),
            None,
        )
        # Dropped spike: Retrieve controller must own the phase.
        if holder is None:
            return base_result

        if self._is_special_phase_result(base_result):
            return base_result

        self._sync_tick_once(game_state)

        target = self.env.targets.get(char.name)
        if target is None:
            # Fallback if real roster name differs from GC shadow name.
            real_attackers = self._real_attackers(game_state)
            try:
                idx = next(i for i, c in enumerate(real_attackers) if c is char)
            except StopIteration:
                idx = -1
            if 0 <= idx < len(self.env.attackers):
                target = self.env.targets.get(self.env.attackers[idx].name)

        if target is None:
            return base_result

        grid = np.asarray(game_state["grid"])
        occupied = {
            tuple(map(int, c.pos))
            for c in game_state.get("chars", [])
            if c is not char
            and bool(getattr(c, "is_alive", True))
        }

        next_pos = _bfs_next_step(
            grid,
            tuple(map(int, char.pos)),
            tuple(map(int, target)),
            occupied,
        )

        # GC phase controllers use (pos, action_type[, payload]).
        return list(next_pos), "MOVE"

    @property
    def current_strategy(self):
        return self.env.current_strategy

    @property
    def macro_targets(self):
        return dict(self.env.targets)
