"""Frozen Omoko Gaming teacher on actual GC characters and public observations."""
import hashlib
from pathlib import Path
from .gc3_teacher_attacker_roster import RosterTeacher


class OmokoTeacher(RosterTeacher):
    def __init__(self):
        from omoko_v1.ov1_attacker_controller import Ov1AttackerController
        root = Path(__file__).resolve().parents[2]/"omoko_v1/data"
        paths = {role:root/f"attacker_{role}_data"/f"dqn_attacker_{role}_best_by_eval.pt"
                 for role in ("carry", "escort", "retrieve", "guard")}
        teacher = Ov1AttackerController(**{role+"_model_path":str(path) for role,path in paths.items()},
                                       greedy=True, device="cpu")
        super().__init__(teacher, "Omoko Gaming")
        self.model_hashes = {role:hashlib.sha256(path.read_bytes()).hexdigest() for role,path in paths.items()}

    @property
    def target_site(self):
        side = self.inner.carry_controller._active_site
        return {"left":0, "right":1}.get(side)

    @property
    def current_map_name(self):
        return "Omoko "+("A" if self.target_site == 0 else "B")

    def decide_move(self, char, state):
        state = dict(state)
        target = self.inner.carry_controller._active_target
        if target is not None:
            # Escorts receive the teacher's actual own plan, not the engine's random marker.
            state["target_plant_pos"] = target
        return super().decide_move(char, state)
