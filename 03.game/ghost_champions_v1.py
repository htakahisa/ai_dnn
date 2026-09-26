from __future__ import annotations
import sys
from pathlib import Path
from controllers import BaseController, DefaultAttackerController, DefaultDefenderController

ROOT = Path(__file__).resolve().parent
GC_DIR = ROOT / "gc_v1"
if str(GC_DIR) not in sys.path:
    sys.path.insert(0, str(GC_DIR))

def _paths(data_dir, *names):
    base = GC_DIR / "data" / data_dir
    return tuple(base / n for n in names)

CARRY = _paths("attacker_carry_gc_data","dqn_attacker_carry_gc_best_by_eval.pt","dqn_attacker_carry_gc_latest.pt")
ESCORT = _paths("attacker_escort_gc_data","dqn_attacker_escort_gc_best_by_eval.pt","dqn_attacker_escort_gc_latest.pt")
RETRIEVE = _paths("attacker_retrieve_gc_data","dqn_attacker_retrieve_gc_best_by_eval.pt","dqn_attacker_retrieve_gc_latest.pt","dqn_attacker_retrieve_gc_final.pt")
GUARD = _paths("attacker_guard_gc_data","dqn_attacker_guard_gc_best_by_eval.pt","dqn_attacker_guard_gc_latest.pt","dqn_attacker_guard_gc_8000_backup.pt")
SEARCH = _paths("defender_search_gc_data","dqn_defender_search_gc_best_by_eval.pt","dqn_defender_search_gc_latest.pt")
RETAKE = _paths("defender_retake_gc_data","dqn_defender_retake_gc_best_by_eval.pt","dqn_defender_retake_gc_final.pt")

def _first_existing(paths):
    return next((p for p in paths if p.exists()), None)

def _load(module_name, class_names, model_paths, greedy=True, **controller_kwargs):
    path = _first_existing(model_paths)
    if path is None:
        print(f"[GC v1][WARN] model missing: {module_name}")
        return None
    try:
        mod = __import__(module_name, fromlist=["*"])
        cls = next((getattr(mod,n,None) for n in class_names if getattr(mod,n,None) is not None), None)
        if cls is None:
            raise AttributeError(f"class not found: {class_names}")
        ctrl = cls(model_path=str(path), greedy=greedy, **controller_kwargs)
        metadata = (f" (positioning_version={getattr(ctrl, 'positioning_version', 0)}, "
                    f"episode={getattr(ctrl, 'model_episode', None)})"
                    if module_name == "learning_attacker_carry_gc" else "")
        print(f"[GC v1] loaded {cls.__name__}: {path}{metadata}")
        return ctrl
    except Exception as e:
        print(f"[GC v1][WARN] load failed {module_name}: {e}")
        return None

class GhostChampionsV1AttackerController(BaseController):
    def __init__(self, greedy=True):
        super().__init__()
        self.fallback = DefaultAttackerController()
        self.carry = _load("learning_attacker_carry_gc",("LearningAttackerCarryGCController","LearningAttackerCarryController"),CARRY,greedy)
        self.escort = _load("learning_attacker_escort_gc",("LearningAttackerEscortGCController","LearningAttackerEscortController"),ESCORT,greedy)
        self.retrieve = _load("learning_attacker_retrieve_gc",("LearningAttackerRetrieveGCController","LearningAttackerRetrieveTouyamaController"),RETRIEVE,greedy)
        self.guard = _load("learning_attacker_guard_gc",("LearningAttackerGuardGCController","LearningAttackerGuardTouyamaController"),GUARD,greedy,verbose=True)
        self.site_ability_used_by_team = False
        print(f"[GC v1][A] carry={self.carry is not None} escort={self.escort is not None} retrieve={self.retrieve is not None} guard={self.guard is not None}")

    def set_game(self, game):
        self.game = game
        # GC攻撃側はAbsolをスパイクキャリアー兼エントリー先頭に固定する。
        # set_game() is also called for the GC defender controller after a
        # side swap. Only force Absol when the GC roster is actually attacking;
        # otherwise this would overwrite the opponent's configured holder.
        attacker_names = {
            str(name) for name in (getattr(game, "attacker_roster", None) or ())
        }
        if "Absol" in attacker_names and hasattr(game, "spike_holder_name"):
            game.spike_holder_name = "Absol"
        self._ensure_absol_carrier()
        for c in (self.fallback,self.carry,self.escort,self.retrieve,self.guard):
            if c is not None and hasattr(c,"set_game"):
                c.set_game(game)

    def _ensure_absol_carrier(self):
        if self.game is None:
            return
        attackers = [
            c for c in getattr(self.game, "chars", [])
            if getattr(c, "team", None) == "A"
        ]
        absol = next((c for c in attackers if c.name == "Absol" and c.is_alive), None)
        carrier = next((c for c in attackers if getattr(c, "has_spike", False)), None)
        if absol is not None and carrier is not None and carrier is not absol:
            carrier.has_spike = False
            absol.has_spike = True

    def reset_round(self):
        self.site_ability_used_by_team = False
        self._ensure_absol_carrier()
        for c in (self.fallback,self.carry,self.escort,self.retrieve,self.guard):
            if c is not None and hasattr(c,"reset_round"):
                c.reset_round()

    def _use(self, ctrl, char, state):
        return (ctrl or self.fallback).decide_move(char,state)

    def _mark_ability(self, result):
        if isinstance(result,tuple) and len(result)>=2 and isinstance(result[1],dict) and result[1].get("ability"):
            self.site_ability_used_by_team = True

    def _cover_result(self, char, game_state, result):
        """孤立したキャリアーに、最寄りのescortを緩やかに寄せる。"""
        if game_state.get("is_planted") or getattr(char, "has_spike", False):
            return result
        if isinstance(result, tuple) and len(result) >= 2:
            if isinstance(result[1], dict) or str(result[1]).upper() in {"PLANT", "ABILITY"}:
                return result

        micro_cover = self._micro_cover_result(char, game_state)
        if micro_cover is not None:
            return micro_cover

        chars = game_state.get("chars", [])
        alive_attackers = [c for c in chars if c.team == "A" and c.is_alive]
        carrier = next((c for c in alive_attackers if getattr(c, "has_spike", False)), None)
        if carrier is None or len(alive_attackers) <= 1:
            return result
        escorts = [c for c in alive_attackers if c is not carrier]
        # Three cells is close enough to trade immediately; six was only
        # visual proximity and still left the carrier effectively isolated.
        support_distance = 3
        supporters = [
            c for c in escorts
            if max(
                abs(c.pos[0] - carrier.pos[0]),
                abs(c.pos[1] - carrier.pos[1]),
            ) <= support_distance
        ]
        # Keep two attackers near the carrier whenever the roster allows it.
        # This remains soft: the escort with the shortest route is nudged in
        # first, and natural combat/ability actions still take precedence.
        required_support = min(2, len(escorts))
        if len(supporters) >= required_support:
            return result
        cover = min(
            [c for c in escorts if c not in supporters] or escorts,
            key=lambda c: max(abs(c.pos[0] - carrier.pos[0]), abs(c.pos[1] - carrier.pos[1])),
        )
        if cover is not char:
            return result

        grid = game_state["grid"]
        occupied = {
            tuple(map(int, c.pos))
            for c in alive_attackers
            if c is not char and c is not carrier
        }
        cr, cc = map(int, carrier.pos)
        plant = game_state.get("target_plant_pos") or carrier.pos
        vr, vc = int(plant[0]) - cr, int(plant[1]) - cc
        preferred = (cr - (1 if vr > 0 else -1 if vr < 0 else 0) * 2,
                     cc - (1 if vc > 0 else -1 if vc < 0 else 0) * 2)
        goals = []
        for r in range(max(0, cr - 3), min(grid.shape[0], cr + 4)):
            for c in range(max(0, cc - 3), min(grid.shape[1], cc + 4)):
                dist = max(abs(r - cr), abs(c - cc))
                if 1 <= dist <= 3 and grid[r, c] != 1 and (r, c) not in occupied:
                    goals.append((r, c))
        if not goals:
            return result
        goal = min(goals, key=lambda p: (max(abs(p[0] - preferred[0]), abs(p[1] - preferred[1])),
                                         max(abs(p[0] - char.pos[0]), abs(p[1] - char.pos[1]))))

        # BFS to the soft cover point, respecting walls and current occupants.
        start = tuple(map(int, char.pos))
        queue = [start]
        parent = {start: None}
        for pos in queue:
            if pos == goal:
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (pos[0] + dr, pos[1] + dc)
                if (nxt in parent or nxt in occupied or
                        not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]) or
                        grid[nxt[0], nxt[1]] == 1):
                    continue
                parent[nxt] = pos
                queue.append(nxt)
        if goal not in parent:
            return result
        step = goal
        while parent[step] is not None and parent[step] != start:
            step = parent[step]
        return [step[0], step[1]]

    def _micro_cover_result(self, char, game_state, max_ticks=5):
        """Nudge a nearby idle teammate into an ongoing crossfire quickly."""
        game = getattr(self, "game", None)
        chars = game_state.get("chars", [])
        if game is None or not char.is_alive:
            return None
        enemies = [c for c in chars if c.is_alive and c.team != char.team]
        allies = [c for c in chars if c.is_alive and c.team == char.team and c is not char]
        engaged = []
        for ally in allies:
            for enemy in enemies:
                if game.check_line_of_sight(ally, enemy):
                    engaged.append((ally, enemy))
        if not engaged:
            return None
        # Do not abandon an active duel; this controller is for the second
        # player who can join the fight.
        if any(game.check_line_of_sight(char, enemy) for _, enemy in engaged):
            return None

        ally, enemy = min(
            engaged,
            key=lambda pair: max(
                abs(char.pos[0] - pair[0].pos[0]),
                abs(char.pos[1] - pair[0].pos[1]),
            ),
        )
        if max(abs(char.pos[0] - ally.pos[0]), abs(char.pos[1] - ally.pos[1])) > max_ticks:
            return None

        grid = game_state["grid"]
        occupied = {tuple(c.pos) for c in chars if c.is_alive and c is not char}
        candidates = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            pos = (int(char.pos[0]) + dr, int(char.pos[1]) + dc)
            if not (0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]):
                continue
            if grid[pos[0], pos[1]] == 1 or pos in occupied:
                continue
            if game.check_cell_line_of_sight(
                pos, tuple(map(int, enemy.pos)), block_smoke=True
            ):
                candidates.append(pos)
        if not candidates:
            return None
        return list(min(candidates, key=lambda p: max(abs(p[0] - ally.pos[0]), abs(p[1] - ally.pos[1]))))

    def decide_move(self, char, game_state):
        if game_state.get("is_planted"):
            return self._use(self.guard,char,game_state)
        if getattr(char,"has_spike",False):
            result = self._use(self.carry,char,game_state)
            self._mark_ability(result)
            return result
        chars = game_state.get("chars",[])
        holder = next((c for c in chars if getattr(c,"is_alive",False) and getattr(c,"team",None)=="A" and getattr(c,"has_spike",False)),None)
        if holder is None:
            return self._use(self.retrieve,char,game_state)
        if self.escort is not None and hasattr(self.escort,"site_ability_used_by_teammate"):
            self.escort.site_ability_used_by_teammate = self.site_ability_used_by_team
        result = self._use(self.escort,char,game_state)
        self._mark_ability(result)
        return self._cover_result(char, game_state, result)

class GhostChampionsV1DefenderController(BaseController):
    def __init__(self, greedy=True):
        super().__init__()
        self.fallback = DefaultDefenderController()
        self.search = _load("learning_defender_search_gc",("LearningDefenderSearchGCController","LearningDefenderSearchTouyamaController"),SEARCH,greedy)
        self.retake = _load("learning_defender_retake_gc",("LearningDefenderRetakeGCController","LearningDefenderRetakeTouyamaController"),RETAKE,greedy)
        print(f"[GC v1][D] search={self.search is not None} retake={self.retake is not None}")

    def set_game(self, game):
        self.game = game
        for c in (self.fallback,self.search,self.retake):
            if c is not None and hasattr(c,"set_game"):
                c.set_game(game)

    def reset_round(self):
        for c in (self.fallback,self.search,self.retake):
            if c is not None and hasattr(c,"reset_round"):
                c.reset_round()

    def decide_move(self, char, game_state):
        ctrl = self.retake if game_state.get("is_planted") else self.search
        return (ctrl or self.fallback).decide_move(char,game_state)

def build_ghost_champions_v1_team_ai():
    from team_ai import DualRoleTeamAI
    return DualRoleTeamAI(
        name="Ghost Champions v1",
        attacker_factory=lambda: GhostChampionsV1AttackerController(greedy=True),
        defender_factory=lambda: GhostChampionsV1DefenderController(greedy=True),
    )
