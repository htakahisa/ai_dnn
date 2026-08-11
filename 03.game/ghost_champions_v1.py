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
GUARD = _paths("attacker_guard_gc_data","dqn_attacker_guard_gc_8000_backup.pt","dqn_attacker_guard_gc_best_by_eval.pt","dqn_attacker_guard_gc_latest.pt")
SEARCH = _paths("defender_search_gc_data","dqn_defender_search_gc_best_by_eval.pt","dqn_defender_search_gc_latest.pt")
RETAKE = _paths("defender_retake_gc_data","dqn_defender_retake_gc_best_by_eval.pt","dqn_defender_retake_gc_final.pt")

def _first_existing(paths):
    return next((p for p in paths if p.exists()), None)

def _load(module_name, class_names, model_paths, greedy=True):
    path = _first_existing(model_paths)
    if path is None:
        print(f"[GC v1][WARN] model missing: {module_name}")
        return None
    try:
        mod = __import__(module_name, fromlist=["*"])
        cls = next((getattr(mod,n,None) for n in class_names if getattr(mod,n,None) is not None), None)
        if cls is None:
            raise AttributeError(f"class not found: {class_names}")
        ctrl = cls(model_path=str(path), greedy=greedy)
        print(f"[GC v1] loaded {cls.__name__}: {path}")
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
        self.guard = _load("learning_attacker_guard_gc",("LearningAttackerGuardGCController","LearningAttackerGuardTouyamaController"),GUARD,greedy)
        self.site_ability_used_by_team = False
        print(f"[GC v1][A] carry={self.carry is not None} escort={self.escort is not None} retrieve={self.retrieve is not None} guard={self.guard is not None}")

    def set_game(self, game):
        self.game = game
        for c in (self.fallback,self.carry,self.escort,self.retrieve,self.guard):
            if c is not None and hasattr(c,"set_game"):
                c.set_game(game)

    def reset_round(self):
        self.site_ability_used_by_team = False
        for c in (self.fallback,self.carry,self.escort,self.retrieve,self.guard):
            if c is not None and hasattr(c,"reset_round"):
                c.reset_round()

    def _use(self, ctrl, char, state):
        return (ctrl or self.fallback).decide_move(char,state)

    def _mark_ability(self, result):
        if isinstance(result,tuple) and len(result)>=2 and isinstance(result[1],dict) and result[1].get("ability"):
            self.site_ability_used_by_team = True

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
        return result

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
