"""Collection-only site curricula; terrain, model weights and combat stay intact.

These temporary selector overrides run in the single-threaded collector process.
They are removed even if a round or serializer raises. Normal evaluation never
enters this context. Saved cases explicitly record the requested site.
"""
from contextlib import contextmanager, ExitStack
import importlib
import random

SAMPLING_VERSION = "site_curriculum_v1"


def pending_site(counts, opponent, sites, target):
    pending = [s for s in sites if counts[opponent, s] < target]
    return min(pending, key=lambda s: (counts[opponent, s], sites.index(s))) if pending else None


@contextmanager
def replace_attribute(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


@contextmanager
def site_sampling(site, *, attacker="touyama_v3"):
    if site is None:
        yield
        return
    if site not in ("L", "R"):
        raise ValueError(f"Unknown collection site: {site}")
    with ExitStack() as stack:
        def replace(owner, name, value):
            stack.enter_context(replace_attribute(owner, name, value))

        if attacker in ("touyama_v3", "toru_ai_v4"):
            module = importlib.import_module("touyama_v3.tv3_learn_attacker_analysis" if attacker == "touyama_v3"
                                             else "toruAI_v4.tv4_learn_attacker_analysis")
            original = module.candidate_routes
            def routes(*args, **kwargs):
                return [r for r in original(*args, **kwargs) if r.site == site]
            replace(module, "candidate_routes", routes)
        elif attacker == "fnatic_v3":
            from fnatic_v3.controller import FnaticV3AttackerController
            def candidates(self, grid):
                return tuple(p for p in self.positions.plant_cells
                             if (p[1] < grid.shape[1] // 2) == (site == "L"))
            def select(self, cells, lengths, grid):
                self.target = random.choice(cells)
                self.attack_region = "A" if site == "L" else "B"
                self.mid_entry_goal, self.mid_entry_done = None, True
                return self.target
            replace(FnaticV3AttackerController, "_plant_candidates", candidates)
            replace(FnaticV3AttackerController, "_select_attack_target", select)
            replace(FnaticV3AttackerController, "_rotate_target", lambda *args: False)
        elif attacker == "gc_v1":
            def cells(game, grid):
                import numpy as np
                return [tuple(map(int, p)) for p in np.argwhere(np.asarray(grid) == 2)
                        if (p[1] < len(grid[0]) // 2) == (site == "L")]
            # The original enforce_attack_target calls these module globals.
            # Replace aliases imported by the actual runtime as well.
            # Load aliases before overriding their source, otherwise a first
            # import could permanently retain the temporary selector.
            modules = [importlib.import_module(name) for name in
                       ("gc_v1.opponent_site_gc", "ghost_champions_v1",
                        "ghost_champions_v1_macro", "gc_v1.learning_attacker_macro_gc_runtime")]
            for module in modules:
                if hasattr(module, "forced_attack_site"):
                    replace(module, "forced_attack_site", lambda game: "A" if site == "L" else "B")
                if hasattr(module, "attack_plant_cells"):
                    replace(module, "attack_plant_cells", cells)
        elif attacker == "omoko_v1":
            # ov1 loads this module under its historic top-level import name.
            import omoko_v1.ov1_attacker_controller
            module = importlib.import_module("ov1_learning_attacker_carry")
            def choose(self, current_round):
                side = "left" if site == "L" else "right"
                return side, random.randrange(max(1, len(self.waypoint_cells_by_site.get(side, ()))))
            replace(module.Ov1LearningAttackerCarryController, "_choose_route", choose)
        elif attacker == "concon_v1":
            from concon_v1 import co1_attacker_controller as module
            from concon_v1.co1_attacker_scenarios import SCENARIOS
            def candidates(report, opponent, map_names, hashes):
                names = tuple(n for n in map_names if SCENARIOS[n].plant_side == ("left" if site == "L" else "right"))
                if not names:
                    raise ValueError(f"ConCon has no registered route for collection site {site}")
                return names
            replace(module, "route_candidates", candidates)
        elif attacker == "frc_v1":
            from frc_v1 import navigation as module
            original_setup, original_navigation = module.attack_setup_positions, module.guard_attack_navigation
            from frc_v1.baseline import plant_sites
            def site_index(snapshot):
                sites = plant_sites(snapshot.grid)
                return next(i for i, cells in enumerate(sites)
                            if (cells[0][1] < len(snapshot.grid[0]) // 2) == (site == "L"))
            def setup(snapshot, index, **kwargs):
                return original_setup(snapshot, site_index(snapshot), **kwargs)
            def navigation(decision, snapshot, masks, **kwargs):
                kwargs["site_index"] = site_index(snapshot)
                return original_navigation(decision, snapshot, masks, **kwargs)
            replace(module, "attack_setup_positions", setup)
            replace(module, "guard_attack_navigation", navigation)
        else:
            raise ValueError(f"No collection site selector for {attacker}")
        yield
