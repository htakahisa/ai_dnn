"""Fixed opponents for collection, training and independent evaluation."""
from pathlib import Path
import hashlib

TORU_BASELINE_OPPONENT = "touyama_v2"
TORU_BEST_DIRECTORY = Path(__file__).resolve().parent.parent / "toruAI_v4" / "data" / "best"
_frozen_hashes = None


def frozen_toru_scenario():
    """Keep the fixed opponent's trained post layout; use corrected LOS at runtime."""
    from toruAI_v4.tv4_scenario import Scenario
    from grid_lines import line_cells
    class FrozenLayout(Scenario):
        def default_posts(self):
            self._building_frozen_posts = True
            try:
                return super().default_posts()
            finally:
                self._building_frozen_posts = False

        def clear(self, origin, target):
            if getattr(self,'_building_frozen_posts',False):
                return all(self.grid[p] != 1 for p in line_cells(origin,target))
            return super().clear(origin,target)
    return FrozenLayout()


def scoped_presets(own, enemy):
    """Keep engine statistics and effects separate for same-name opponents."""
    from dataclasses import replace
    from run_competition_manager import TeamPlayerKey
    return (replace(own, players=tuple(TeamPlayerKey(n, 'touyama-v3:own') for n in own.players)),
            replace(enemy, players=tuple(TeamPlayerKey(n, 'touyama-v3:opponent') for n in enemy.players)))


def toru_baseline_hashes():
    names = ("defender_analysis_best.pt", "search_best.pt", "retake_L_best.pt",
             "retake_R_best.pt", "attacker_analysis_best.pt", "attacker_plant_best.pt",
             "attacker_guard_best.pt")
    return {name: hashlib.sha256((TORU_BEST_DIRECTORY / TORU_BASELINE_OPPONENT / name).read_bytes()).hexdigest()
            for name in names}


def build_opponent_ai(key, device="cpu"):
    if key != "toru_ai_v4":
        from run_game import _build_team_ai
        return _build_team_ai(key, device=device)
    global _frozen_hashes
    current = toru_baseline_hashes()
    if _frozen_hashes is not None and current != _frozen_hashes:
        raise ValueError("Toru v4 baseline changed during this run; restart with consistent source models")
    _frozen_hashes = current
    from team_ai import DualRoleTeamAI
    from toruAI_v4.tv4_defender_controller import ToruV4DefenderController
    from toruAI_v4.tv4_attacker_guard_controller import ToruV4AttackerPlantGuardController
    def attacker():
        return ToruV4AttackerPlantGuardController.from_best(
            TORU_BASELINE_OPPONENT, best_dir=TORU_BEST_DIRECTORY,scenario=frozen_toru_scenario())
    def defender():
        return ToruV4DefenderController(TORU_BASELINE_OPPONENT,
            scenario=frozen_toru_scenario(),
            search_path=TORU_BEST_DIRECTORY / TORU_BASELINE_OPPONENT / "search_best.pt",
            retake_path=TORU_BEST_DIRECTORY)
    return DualRoleTeamAI("Toru AI v4", attacker, defender)
