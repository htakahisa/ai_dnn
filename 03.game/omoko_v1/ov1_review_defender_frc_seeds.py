"""Evaluate defender recon overrides with fixed weights; never train models."""

from collections import Counter
from contextlib import ExitStack
import json
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

AI_DIR = Path(__file__).resolve().parent
ROOT_DIR = AI_DIR.parent
SEEDS = (3338407445, 1596261721)
VARIANTS = ("baseline", "current_defender_recon")
MAP_REVISION = "13c1414000f7f12f78470c3536f9a45041bc1e92"
DEFENDER_MODEL_REVISION = MAP_REVISION
OUTPUT_PATH = AI_DIR / "logs" / "ov1_defender_frc_seed_review.json"
MAX_OUTPUT_BYTES = 1_000_000

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(AI_DIR))


def main():
    # Restore only the match-time deployment string in memory. Preserve the
    # user's uncommitted deployment edits and all model files on disk.
    import ov1_map_data_search as map_module
    original = subprocess.check_output(
        ["git", "show", f"{MAP_REVISION}:03.game/omoko_v1/ov1_map_data_search.py"],
        cwd=ROOT_DIR,
    ).decode("utf-8")
    namespace = {}
    exec(compile(original, "<match-time-search-map>", "exec"), namespace)
    map_module.SEARCH_MAZE_STR = namespace["SEARCH_MAZE_STR"]

    from party_presets import get_preset
    from run_competition_manager import play_map
    import omoko_v1.ov1_defender_controller as defender

    real_recon = defender.recon_response
    import torch
    frozen_models, model_hashes = {}, {}
    for role in ("search", "retake"):
        model_file = f"omoko_v1/data/defender_{role}_data/dqn_defender_{role}_best_by_eval.pt"
        raw = subprocess.check_output(
            ["git", "show", f"{DEFENDER_MODEL_REVISION}:03.game/{model_file}"],
            cwd=ROOT_DIR,
        )
        frozen_models[role] = torch.load(io.BytesIO(raw), map_location="cpu")
        model_hashes[role] = hashlib.sha256(raw).hexdigest()
    prior_source = subprocess.check_output(
        ["git", "show", f"{MAP_REVISION}:03.game/omoko_v1/ov1_inference_recon_response.py"],
        cwd=ROOT_DIR,
    ).decode("utf-8")
    prior_namespace = {}
    exec(compile(prior_source, "<prior-defender-recon>", "exec"), prior_namespace)
    prior_recon = prior_namespace["recon_response"]
    real_init = defender.Ov1DefenderController.__init__

    def init_with_absolute_models(self, *args, **kwargs):
        kwargs.setdefault("search_model_path", str(AI_DIR / "data/defender_search_data/dqn_defender_search_best_by_eval.pt"))
        kwargs.setdefault("retake_model_path", str(AI_DIR / "data/defender_retake_data/dqn_defender_retake_best_by_eval.pt"))
        real_init(self, *args, **kwargs)
        self.search_controller.model.load_state_dict(frozen_models["search"])
        self.retake_controller.model.load_state_dict(frozen_models["retake"])
    report = {"map_revision": MAP_REVISION, "defender_model_revision": DEFENDER_MODEL_REVISION,
              "defender_model_hashes": model_hashes, "seeds": SEEDS, "runs": []}
    for variant in VARIANTS:
        fatigue, tactical = {}, {}
        series_wins = [0, 0]
        for map_number, seed in enumerate(SEEDS, 1):
            calls = Counter()

            def response(char, state, memory):
                if variant == "without_defender_recon" or (
                    variant == "without_retake_recon" and state.get("is_planted")
                ):
                    return None
                handler = real_recon if variant == "current_defender_recon" else prior_recon
                result = handler(char, state, memory)
                if result is not None:
                    calls["retake" if state.get("is_planted") else "search"] += 1
                    calls["revealed" if getattr(char, "reveal_remaining", 0) > 0 else "shoot"] += 1
                return result

            with ExitStack() as stack:
                stack.enter_context(patch.object(defender.Ov1DefenderController, "__init__", init_with_absolute_models))
                stack.enter_context(patch.object(defender, "recon_response", response))
                result = play_map(
                    get_preset("Omoko Gaming"), get_preset("Furina Classic"),
                    map_number, seed, False,
                    team1_controller_key="omoko_gaming_v1",
                    team2_controller_key="frc_v1",
                    team1_series_wins=series_wins[0],
                    team2_series_wins=series_wins[1], series_maps_to_win=2,
                    mental_fatigue_state=fatigue, series_tactical_state=tactical,
                )
            series_wins[0 if result.score1 > result.score2 else 1] += 1
            sides = {}
            for side in ("attacker", "defender"):
                rounds = [r for r in result.round_records if any(
                    p.get("team") == "Omoko Gaming" and p.get("side") == side
                    for p in r["players"].values())]
                sides[side] = dict(
                    rounds=len(rounds), wins=sum(r["winner"] == side for r in rounds),
                    plants=sum(r["planted"] for r in rounds),
                    losses_by_reason=dict(Counter(r["reason"] for r in rounds if r["winner"] != side)),
                )
            run = dict(variant=variant, seed=seed, score=[result.score1, result.score2],
                       sides=sides, recon_overrides=dict(calls))
            report["runs"].append(run)
            print(json.dumps(run, ensure_ascii=False), flush=True)
            encoded = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
            if len(encoded) > MAX_OUTPUT_BYTES:
                raise RuntimeError("Review report exceeded its pre-write size limit")
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_bytes(encoded)


if __name__ == "__main__":
    main()
