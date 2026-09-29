"""Match-local referee audit of coach normal-ability outcomes.

This instrument attaches only to a headless evaluation game. Actors and their
observations never read its events. It wraps existing methods and preserves
their return values and game state.
"""

from __future__ import annotations

from typing import Any


class AbilityEffectAudit:
    def __init__(self, game, coach_team_ai, *, smoke_measure=None) -> None:
        self.game = game
        self.coach_team_ai = coach_team_ai
        self.events: list[dict[str, Any]] = []
        self._projectiles: dict[int, dict[str, Any]] = {}
        self._smokes: dict[int, tuple[dict[str, Any], dict]] = {}
        self._smoke_measure = smoke_measure

    def install(self) -> "AbilityEffectAudit":
        game = self.game
        original_ability = game.execute_ai_ability
        original_flash = game._explode_flash
        original_recon = game._explode_recon
        original_frame = game._record_replay_frame

        def execute(owner, action):
            coach_team = "A" if game.current_attacker_team_ai is self.coach_team_ai else "D"
            relevant = owner.team == coach_team
            before_flash = len(game.flash_projectiles)
            before_recon = len(game.recon_projectiles)
            before_smoke = len(game.smokes)
            success = original_ability(owner, action)
            if not relevant or not isinstance(action, dict):
                return success
            ability = str(action.get("ability", "")).upper()
            if ability not in {"SMOKE", "FLASH", "RECON"}:
                return success
            event = {
                "round": int(game.current_round), "tick": int(game.battle_tick),
                "side": "attacker" if coach_team == "A" else "defender",
                "owner": str(owner.name), "ability": ability,
                "target": list(action["target"]) if isinstance(action.get("target"), (list, tuple)) else None,
                "cast_success": bool(success), "resolved": not bool(success),
                "affected_enemies": [], "enemy_lane_pair_ticks_blocked": 0,
                "ally_lane_pair_ticks_blocked": 0,
            }
            self.events.append(event)
            if success and ability == "FLASH" and len(game.flash_projectiles) > before_flash:
                self._projectiles[id(game.flash_projectiles[-1])] = event
            elif success and ability == "RECON" and len(game.recon_projectiles) > before_recon:
                self._projectiles[id(game.recon_projectiles[-1])] = event
            elif success and ability == "SMOKE" and len(game.smokes) > before_smoke:
                self._smokes[id(game.smokes[-1])] = (event, game.smokes[-1])
            return success

        def explode(original, projectile, status, *args, **kwargs):
            event = self._projectiles.pop(id(projectile), None)
            before = {id(char): int(getattr(char, status, 0)) for char in game.chars}
            result = original(projectile, *args, **kwargs)
            if event is not None:
                affected = [str(char.name) for char in game.chars
                            if char.is_alive and char.team != projectile.get("team")
                            and int(getattr(char, status, 0)) > before[id(char)]]
                event["affected_enemies"] = sorted(affected)
                event["resolved"] = True
            return result

        def explode_flash(projectile, *args, **kwargs):
            return explode(original_flash, projectile, "blind_remaining", *args, **kwargs)

        def explode_recon(projectile, *args, **kwargs):
            return explode(original_recon, projectile, "reveal_remaining", *args, **kwargs)

        def record_frame():
            result = original_frame()
            if self._smokes:
                measure = self._smoke_measure
                if measure is None:
                    from coach_v1.evaluate_task09a_gorimaru_smoke import marginal_shot_lanes
                    measure = marginal_shot_lanes
                for key, (event, smoke) in tuple(self._smokes.items()):
                    if not any(item is smoke for item in game.smokes) or int(game.current_round) != event["round"]:
                        event["resolved"] = True
                        self._smokes.pop(key, None)
                        continue
                    team = "A" if event["side"] == "attacker" else "D"
                    lanes = measure(game, team, smoke)
                    event["enemy_lane_pair_ticks_blocked"] += int(lanes["enemy_lanes_blocked"])
                    event["ally_lane_pair_ticks_blocked"] += int(lanes["ally_lanes_blocked"])
                    if game.replay_frames[-1].get("round_over"):
                        event["resolved"] = True
                        self._smokes.pop(key, None)
            return result

        game.execute_ai_ability = execute
        game._explode_flash = explode_flash
        game._explode_recon = explode_recon
        game._record_replay_frame = record_frame
        return self

    def summary(self) -> dict[str, dict[str, int | float | None]]:
        """Per-ability effective rate among successful, resolved casts only."""
        return summarize_ability_events(self.events)


def summarize_ability_events(events) -> dict[str, dict[str, int | float | None]]:
    """Per-ability effective rate among successful, resolved casts only."""
    result = {}
    for ability in ("SMOKE", "FLASH", "RECON"):
        selected = [item for item in events if item["ability"] == ability]
        casts = [item for item in selected if item["cast_success"]]
        resolved = [item for item in casts if item["resolved"]]
        effective = sum(bool(item["affected_enemies"]) if ability != "SMOKE"
                        else item["enemy_lane_pair_ticks_blocked"] > item["ally_lane_pair_ticks_blocked"]
                        for item in resolved)
        result[ability] = {
            "requests": len(selected), "successful_casts": len(casts),
            "resolved_casts": len(resolved), "effective_casts": effective,
            "effective_rate": effective / len(resolved) if resolved else None,
        }
    return result
