from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from controllers import DefaultDefenderController
from defender_policy_common import (
    ABILITY_BY_ACTION,
    DIRECTION_BY_ACTION,
    action_mask,
    alive,
    defender_observation_to_vector,
    get_defender_observation,
    load_defender_policy,
    masked_probs,
    valid_destination,
)


class PolicyDefenderController:
    def __init__(
        self,
        model_path: str | Path = "policy_fnatic_defender_dagger_final.pt",
        device: str = "auto",
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDAが利用できません")

        self.model, self.obs_size = load_defender_policy(
            model_path,
            self.device,
        )
        self.game = None
        self.helper = DefaultDefenderController()
        print(f"Fnatic defender model: {model_path} ({self.device})")

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        if hasattr(self.helper, "reset_round"):
            self.helper.reset_round()

    @staticmethod
    def _closest_enemy(char: Any, game_state: dict):
        enemies = [
            other
            for other in game_state.get("chars", [])
            if alive(other) and getattr(other, "team", None) != char.team
        ]
        if not enemies:
            return None
        return min(
            enemies,
            key=lambda enemy: max(
                abs(int(enemy.pos[0]) - int(char.pos[0])),
                abs(int(enemy.pos[1]) - int(char.pos[1])),
            ),
        )

    def _ability_target(
        self,
        ability_name: str,
        char: Any,
        game_state: dict,
        helper_result: Any,
    ) -> tuple[int, int]:
        # 教師補助が同じアビリティを選んだ場合は、その標的を優先する。
        if (
            isinstance(helper_result, tuple)
            and len(helper_result) == 2
            and isinstance(helper_result[1], dict)
            and str(helper_result[1].get("ability", "")).upper() == ability_name
        ):
            target = helper_result[1].get("target")
            if target is not None:
                return int(target[0]), int(target[1])

        if ability_name == "RECON":
            target = game_state.get("planted_pos") or game_state.get("target_plant_pos")
            if target is not None:
                return int(target[0]), int(target[1])

        closest_enemy = self._closest_enemy(char, game_state)
        if closest_enemy is not None:
            return int(closest_enemy.pos[0]), int(closest_enemy.pos[1])

        return int(char.pos[0]), int(char.pos[1])

    def decide_move(self, char, game_state):
        if self.game is None:
            raise RuntimeError("set_game(game)が未実行です")

        observation = get_defender_observation(self.game, char)
        vector = defender_observation_to_vector(observation)
        if len(vector) != self.obs_size:
            raise ValueError(f"obs mismatch {len(vector)} != {self.obs_size}")

        with torch.no_grad():
            tensor = torch.from_numpy(vector).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probabilities = masked_probs(
                logits,
                action_mask(self.game, char, game_state),
            )
            action_idx = int(probabilities.argmax(1).item())

        row, col = map(int, char.pos)

        if action_idx in DIRECTION_BY_ACTION:
            dr, dc = DIRECTION_BY_ACTION[action_idx]
            nr, nc = row + dr, col + dc
            if valid_destination(self.game, char, nr, nc):
                return [nr, nc]
            return [row, col]

        if action_idx in ABILITY_BY_ACTION:
            ability_name = ABILITY_BY_ACTION[action_idx]
            charge_name = {
                "SMOKE": "smoke_charges",
                "FLASH": "flash_charges",
                "RECON": "recon_charges",
            }[ability_name]

            if getattr(char, charge_name, 0) <= 0:
                return [row, col]

            helper_result = self.helper.decide_move(char, game_state)
            target = self._ability_target(
                ability_name,
                char,
                game_state,
                helper_result,
            )
            return [row, col], {
                "ability": ability_name,
                "target": target,
            }

        if action_idx == 8:
            planted_pos = game_state.get("planted_pos")
            if (
                planted_pos is not None
                and game_state.get("is_planted", False)
                and max(
                    abs(int(planted_pos[0]) - row),
                    abs(int(planted_pos[1]) - col),
                )
                <= 1
            ):
                return [row, col], "DEFUSE"

        return [row, col]
