"""
学習済みBC / DAggerモデルをアタッカー側に使い、試合を画面表示する。

ディフェンダー側は既存のルールベースAI。
観測形式・9アクション・アクションマスク・設置/回収補助は evaluate_bc_dagger.py と共通。

使い方:
    python watch_policy_battle.py
    python watch_policy_battle.py --model policy_dagger_final.pt
    python watch_policy_battle.py --device cuda
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from controllers import DefaultDefenderController
from evaluate_bc_dagger import EvaluationAttackerController, load_policy
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="学習済みモデルの試合をTkinter画面で観戦します。"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("policy_bc_final.pt"),
        help="表示に使うモデルファイル（既定: policy_bc_final.pt）",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="推論デバイス（既定: auto）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="編成や戦闘乱数をある程度固定したい場合の乱数シード",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できないため --device cuda は使用できません。")
    return requested


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {model_path}\n"
            "--model で正しい .pt ファイルを指定してください。"
        )

    device = resolve_device(args.device)
    model, obs_size = load_policy(model_path, device)

    attacker_controller = EvaluationAttackerController(
        model=model,
        obs_size=obs_size,
        device=device,
    )
    defender_controller = DefaultDefenderController()

    attacker_roster, defender_roster = build_two_balanced_rosters()

    print(f"Model            : {model_path.name}")
    print(f"Device           : {device}")
    print(f"Attacker roster  : {attacker_roster}")
    print(f"Defender roster  : {defender_roster}")
    print("Attacker control : learned policy")
    print("Defender control : rule-based AI")

    game = VisualFPSBattle(
        NEW_MAZE_STR,
        attacker_controller,
        defender_controller,
        headless=False,
        disable_side_swap=True,
        attacker_roster=attacker_roster,
        defender_roster=defender_roster,
    )

    # 学習済みControllerが観測生成とアクションマスクにgame本体を使う。
    attacker_controller.set_game(game)

    game.run()


if __name__ == "__main__":
    main()
