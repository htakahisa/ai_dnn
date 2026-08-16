"""Runtime for Ghost Champions Defender Opening Macro.

Place in:
    gc_v1/learning_defender_opening_macro_gc_runtime.py

Expected companions in gc_v1/:
    learning_defender_opening_macro_gc.py
    defender_opening_ability_patterns_gc.py
    map_data_defender_opening_ability_gc.py

Checkpoint priority:
    data/defender_opening_macro_gc_data/
        dqn_defender_opening_macro_gc_best.pt
        dqn_defender_opening_macro_gc_latest.pt
        dqn_defender_opening_macro_gc_final.pt

The combined checkpoint is produced by train_defender_opening_macro_gc.py.
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from learning_defender_opening_macro_gc import (
    ABILITY_ORDER,
    OBS_DIM,
    EXEC_ACTION_DIM,
    EXEC_WAIT,
    EXECUTE,
    EXEC_CANCEL,
    LearningDefenderOpeningMacroGCController as _BaseOpeningMacro,
)


class OpeningSelectionQNet(nn.Module):
    """Same architecture as train_defender_opening_macro_gc.py."""

    def __init__(self, obs_dim: int, action_dims: dict[str, int]):
        super().__init__()
        self.action_dims = dict(action_dims)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            ability: nn.Linear(128, int(dim))
            for ability, dim in self.action_dims.items()
        })

    def forward(self, x):
        h = self.trunk(x)
        return {ability: head(h) for ability, head in self.heads.items()}


class OpeningExecutionQNet(nn.Module):
    """Same architecture as train_defender_opening_macro_gc.py."""

    def __init__(self, obs_dim=OBS_DIM, action_dim=EXEC_ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def _default_model_candidates():
    here = Path(__file__).resolve().parent

    # Normal placement: 03.game/gc_v1/this_file.py
    if here.name == "gc_v1":
        data_dir = here / "data" / "defender_opening_macro_gc_data"
    else:
        # Also tolerate running the file from 03.game root during testing.
        data_dir = here / "gc_v1" / "data" / "defender_opening_macro_gc_data"

    return (
        data_dir / "dqn_defender_opening_macro_gc_best.pt",
        data_dir / "dqn_defender_opening_macro_gc_latest.pt",
        data_dir / "dqn_defender_opening_macro_gc_final.pt",
    )


def find_opening_macro_model():
    for path in _default_model_candidates():
        if path.is_file():
            return path
    return None


class LearningDefenderOpeningMacroGCRuntime(_BaseOpeningMacro):
    """Greedy real-match runtime for the trained Opening Macro."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        greedy: bool = True,
        verbose: bool = False,
        seed: int | None = None,
    ):
        # Base class owns pattern parsing, BFS movement and emergency cancellation.
        super().__init__(
            model_path=None,
            greedy=greedy,
            verbose=verbose,
            seed=seed,
        )

        self.device = torch.device("cpu")
        self.runtime_greedy = bool(greedy)
        self.rng = random.Random(seed)

        # Runtime-only set-play variation.
        # Pattern selection remains learned; setup/timing get small random variation.
        self._runtime_ready_tick = {}
        self._runtime_delay = {}
        self._runtime_current_ability = None
        self._runtime_variation_pending = True

        resolved = Path(model_path) if model_path is not None else find_opening_macro_model()
        if resolved is None:
            raise FileNotFoundError(
                "Defender Opening Macro checkpoint not found. Expected one of: "
                + ", ".join(str(p) for p in _default_model_candidates())
            )

        self.model_path = str(resolved)
        checkpoint = torch.load(
            resolved,
            map_location=self.device,
            weights_only=False,
        )

        if int(checkpoint.get("obs_dim", -1)) != OBS_DIM:
            raise ValueError(
                f"Opening Macro OBS_DIM mismatch: "
                f"checkpoint={checkpoint.get('obs_dim')} runtime={OBS_DIM}"
            )

        self.action_dims = {
            str(k): int(v)
            for k, v in checkpoint["action_dims"].items()
        }

        # Validate checkpoint heads against the currently configured map.
        expected_dims = {
            ability: 1 + len(self.patterns.get(ability, {}))
            for ability in ABILITY_ORDER
        }
        if self.action_dims != expected_dims:
            raise ValueError(
                "Opening Macro pattern/action mismatch. "
                f"checkpoint={self.action_dims}, current_map={expected_dims}. "
                "The ability map changed after training."
            )

        if int(checkpoint.get("execution_action_dim", -1)) != EXEC_ACTION_DIM:
            raise ValueError(
                "Opening Macro execution action dimension mismatch: "
                f"checkpoint={checkpoint.get('execution_action_dim')} "
                f"runtime={EXEC_ACTION_DIM}"
            )

        self.selection_net = OpeningSelectionQNet(
            OBS_DIM,
            self.action_dims,
        ).to(self.device)
        self.execution_net = OpeningExecutionQNet(
            OBS_DIM,
            EXEC_ACTION_DIM,
        ).to(self.device)

        self.selection_net.load_state_dict(
            checkpoint["selection_state_dict"]
        )
        self.execution_net.load_state_dict(
            checkpoint["execution_state_dict"]
        )

        self.selection_net.eval()
        self.execution_net.eval()

        self.selection_catalog = {
            ability: [None] + sorted(self.patterns.get(ability, {}).keys())
            for ability in ABILITY_ORDER
        }

        self.checkpoint_episode = int(checkpoint.get("episode", 0))
        self.checkpoint_best_win_rate = float(
            checkpoint.get("best_win_rate", -1.0)
        )

        if self.verbose:
            print(
                "[GC D-OPENING] loaded "
                f"{resolved} episode={self.checkpoint_episode} "
                f"bestWR={self.checkpoint_best_win_rate:.3f}"
            )

    def reset_round(self):
        super().reset_round()
        self._runtime_selection_done = False
        self._runtime_selection_actions = {}
        self._runtime_ready_tick = {}
        self._runtime_delay = {}
        self._runtime_current_ability = None
        self._runtime_variation_pending = True

    def _apply_random_setup_variation(self):
        """Choose setup variation once per round.

        The setup controller reads gc_setup_variation_index when that integration
        is present. Keeping this here makes real matches vary like the watch build.
        """
        if not self._runtime_variation_pending or self.game is None:
            return
        idx = self.rng.randrange(3)
        self.game.gc_setup_variation_index = idx
        self.game.gc_setup_variation = ("BALANCED", "LEFT_LEAN", "RIGHT_LEAN")[idx]
        self._runtime_variation_pending = False
        if self.verbose:
            print(f"[GC D-OPENING] setup variation={self.game.gc_setup_variation}")

    def _delay_for_ability(self, ability):
        ability = str(ability or "").upper()
        if ability not in self._runtime_delay:
            if ability == "SMOKE":
                delay = self.rng.randint(0, 2)
            elif ability == "RECON":
                delay = self.rng.randint(4, 10)
            elif ability == "FLASH":
                delay = self.rng.randint(7, 14)
            else:
                delay = 0
            self._runtime_delay[ability] = int(delay)
        return self._runtime_delay[ability]

    def _select_from_q(self, q):
        q = np.asarray(q, dtype=np.float32)
        if self.runtime_greedy:
            return int(np.argmax(q))

        # Optional stochastic inference. Normal GC runtime uses greedy=True.
        probs = np.exp(q - np.max(q))
        probs = probs / max(float(probs.sum()), 1e-8)
        return int(self.rng.choices(
            range(len(q)),
            weights=probs.tolist(),
            k=1,
        )[0])

    def _initialize_runtime_selection(self, game_state):
        """Select NONE/pattern independently for Smoke, Flash and Recon once."""
        if self._runtime_selection_done:
            return

        self._runtime_selection_done = True
        obs = self.build_observation(game_state)
        x = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_by_ability = self.selection_net(x)

        for ability in ABILITY_ORDER:
            q = q_by_ability[ability][0].cpu().numpy()
            action = self._select_from_q(q)
            catalog = self.selection_catalog[ability]

            if not (0 <= action < len(catalog)):
                continue

            pattern_id = catalog[action]
            self._runtime_selection_actions[ability] = (
                None if pattern_id is None else int(pattern_id)
            )

            if pattern_id is None:
                if self.verbose:
                    print(f"[GC D-OPENING] {ability}: NONE")
                continue

            ok = self.set_plan(
                ability,
                int(pattern_id),
                game_state,
            )

            if self.verbose and not ok:
                print(
                    f"[GC D-OPENING] {ability}#{pattern_id}: "
                    "selected but currently infeasible"
                )

    def _policy_execution_action(self, game_state):
        """Runtime timing variation.

        Selection is still learned by selection_net.  Execution timing is treated
        as harmless set-play variation rather than a deterministic greedy choice.
        Emergency cancellation remains handled by the base coordinator.
        """
        ability = str(self._runtime_current_ability or "").upper()
        now = int(self.tick)

        if ability not in self._runtime_ready_tick:
            self._runtime_ready_tick[ability] = now

        ready_tick = int(self._runtime_ready_tick[ability])
        delay = self._delay_for_ability(ability)

        if now < ready_tick + delay:
            return EXEC_WAIT
        return EXECUTE


    def coordinate(self, char, game_state, base_result):
        # Plant ends the Opening phase immediately.
        if bool(game_state.get("is_planted", False)):
            return base_result

        self._apply_random_setup_variation()
        self._initialize_runtime_selection(game_state)

        # Tell _policy_execution_action which ability's READY decision this is.
        self._runtime_current_ability = None
        char_name = str(getattr(char, "name", ""))
        for ability, plan in self.plans.items():
            if (
                getattr(plan, "active", False)
                and str(getattr(plan, "caster_name", "")) == char_name
            ):
                self._runtime_current_ability = str(ability).upper()
                break

        try:
            return super().coordinate(
                char,
                game_state,
                base_result,
            )
        finally:
            self._runtime_current_ability = None


# Short alias for wrapper code.
DefenderOpeningMacroGCRuntime = LearningDefenderOpeningMacroGCRuntime
