"""
Behavioral Cloning（模倣学習）
ルールベースAIのデモデータからニューラルネットを訓練
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    """ポリシーネットワーク：状態 → アクション確率"""

    def __init__(self, obs_size=2672, num_actions=13):
        """
        obs_size: 観測ベクトルのサイズ
        num_actions: 可能なアクション数
                    (移動8方向 + スモーク + フラッシュ + リコン + その他) = 13
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),  # 出力層：各アクションのロジット
        )

    def forward(self, obs):
        """
        Args:
            obs: (batch_size, obs_size) のテンソル
        Returns:
            logits: (batch_size, num_actions) のテンソル
        """
        return self.net(obs)


class BCTrainer:
    """模倣学習トレーナー"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.train_losses = []
        self.val_losses = []

    def load_demos(self, demo_file):
        """JSONデモファイルからデータを読み込む"""
        print(f"デモファイル読み込み：{demo_file}")

        with open(demo_file, "r") as f:
            demos = json.load(f)

        print(f"読み込んだデモ：{len(demos)} ペア")

        # 観測を数値ベクトルに変換
        obs_list = []
        action_list = []

        for demo in demos:
            obs = demo["observation"]
            action = demo["action"]

            # 今回はアタッカー側のみを学習対象にする
            # （ディフェンダー側の action は、observation が
            #   「アタッカー視点」で記録されているため学習データとして使えない）
            if action.get("team") != "A":
                continue

            # グリッドをフラット化
            grid_vec = np.array(obs["grid"], dtype=np.float32)

            # 味方情報：5人 × 6要素 = 30
            ally_vec = np.zeros(30, dtype=np.float32)
            for i, ally in enumerate(obs["allies"][:5]):
                rel_pos = ally.get("rel_pos", ally.get("pos", [0, 0]))
                ally_vec[i * 6 : (i + 1) * 6] = [
                    rel_pos[0] / 25.0,
                    rel_pos[1] / 35.0,
                    ally["hp"] / 100.0,
                    float(ally.get("has_spike", 0)),
                    float(ally["recon_cd"]),
                    float(ally["flash_cd"]),
                ]

            # 敵情報：5人 × 3要素 = 15
            enemy_vec = np.zeros(15, dtype=np.float32)
            for i, enemy in enumerate(obs["visible_enemies"][:5]):
                enemy_vec[i * 3 : (i + 1) * 3] = [
                    enemy["rel_pos"][0] / 25.0,
                    enemy["rel_pos"][1] / 35.0,
                    enemy["hp"] / 100.0,
                ]

            # ゲーム状態
            game_state_vec = np.array(obs["game_state"], dtype=np.float32)

            # 落ちているスパイク位置
            spike_pos = obs.get("spike_pos", [0, 0])
            spike_vec = np.array(
                [
                    spike_pos[0] / 25.0,
                    spike_pos[1] / 35.0,
                ],
                dtype=np.float32,
            )

            # プラント目標地点
            target_plant_pos = obs.get("target_plant_pos", [0, 0])
            plant_target_vec = np.array(
                [
                    target_plant_pos[0] / 25.0,
                    target_plant_pos[1] / 35.0,
                ],
                dtype=np.float32,
            )

            # 見えている敵人数：0〜5人を0〜1に正規化
            visible_enemy_count = float(
                obs.get("visible_enemy_count", len(obs.get("visible_enemies", [])))
            )
            visible_enemy_count_vec = np.array(
                [min(visible_enemy_count, 5.0) / 5.0],
                dtype=np.float32,
            )

            # サイトまでのマンハッタン距離を0〜1に正規化
            distance_to_site = float(obs.get("distance_to_site", 0.0))
            distance_to_site_vec = np.array(
                [min(distance_to_site, 60.0) / 60.0],
                dtype=np.float32,
            )

            # 全て結合
            full_obs = np.concatenate(
                [
                    grid_vec,
                    ally_vec,
                    enemy_vec,
                    game_state_vec,
                    spike_vec,
                    plant_target_vec,
                    visible_enemy_count_vec,
                    distance_to_site_vec,
                ]
            )

            obs_list.append(full_obs)

            # アクションを離散値に変換（自キャラの現在地をobsから探して方向を計算）
            action_idx = self._encode_action(action, obs)
            action_list.append(action_idx)

        obs_array = np.array(obs_list, dtype=np.float32)
        action_array = np.array(action_list, dtype=np.int64)

        print(f"観測形状：{obs_array.shape}")
        print(f"アクション形状：{action_array.shape}")
        print(
            f"（アタッカー側のみに絞り込み: {len(obs_array)} / {len(demos)} 件を使用）"
        )

        return obs_array, action_array

    def _encode_action(self, action_data, obs):
        """
        アクションデータを離散的なアクション番号に変換

        0-7: 移動方向（上, 下, 左, 右, 左上, 右上, 左下, 右下）
        8: スモーク
        9: フラッシュ
        10: リコン
        11: 停止（移動なし）
        12: プラント
        """
        ability = action_data.get("ability")
        if ability == "SMOKE":
            return 8
        elif ability == "FLASH":
            return 9
        elif ability == "RECON":
            return 10

        # 移動方向を計算：obsの中から自分自身の現在地を探す
        char_name = action_data.get("char")
        move_to = action_data.get("move", [0, 0])

        current_pos = None
        for ally in obs["allies"]:
            if ally.get("name") == char_name:
                current_pos = ally["pos"]
                break

        if current_pos is None:
            return 11  # 自分の情報が見つからない場合は「停止」扱い

        special = action_data.get("special")
        if special == "PLANT":
            return 12

        dr = move_to[0] - current_pos[0]
        dc = move_to[1] - current_pos[1]

        # 8方向マッピング（controllers.pyのdirectionsと対応）
        direction_map = {
            (-1, 0): 0,  # 上
            (1, 0): 1,  # 下
            (0, -1): 2,  # 左
            (0, 1): 3,  # 右
            (-1, -1): 4,  # 左上
            (-1, 1): 5,  # 右上
            (1, -1): 6,  # 左下
            (1, 1): 7,  # 右下
        }

        # dr, dc を -1/0/1 に正規化（BFS移動は1マスずつのはずだが念のため）
        norm_dr = max(-1, min(1, dr))
        norm_dc = max(-1, min(1, dc))

        if (norm_dr, norm_dc) == (0, 0):
            return 11  # 停止

        return direction_map.get((norm_dr, norm_dc), 11)

    def train(self, obs_array, action_array, epochs=50, batch_size=32, val_split=0.1):
        """
        模倣学習を実行

        Args:
            obs_array: (N, obs_size) の観測配列
            action_array: (N,) のアクション配列
            epochs: エポック数
            batch_size: バッチサイズ
            val_split: 検証データの割合
        """
        print(f"\n【模倣学習開始】")
        print(f"  エポック：{epochs}")
        print(f"  バッチサイズ：{batch_size}")

        self.train_losses = []
        self.val_losses = []

        # テンソル化
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        action_tensor = torch.LongTensor(action_array).to(self.device)

        # 訓練/検証分割
        n = len(obs_array)
        val_size = int(n * val_split)
        indices = np.random.permutation(n)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        obs_train = obs_tensor[train_indices]
        action_train = action_tensor[train_indices]
        obs_val = obs_tensor[val_indices]
        action_val = action_tensor[val_indices]

        # Dataset & DataLoader
        train_dataset = TensorDataset(obs_train, action_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # モデル初期化
        obs_size = obs_array.shape[1]
        self.model = PolicyNetwork(obs_size=obs_size, num_actions=13).to(self.device)

        # 最適化器と損失関数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        loss_fn = nn.CrossEntropyLoss()

        # 訓練ループ
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # 訓練
            self.model.train()
            train_loss = 0.0
            for obs_batch, action_batch in train_loader:
                logits = self.model(obs_batch)
                loss = loss_fn(logits, action_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # 検証
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(obs_val)
                val_loss = loss_fn(val_logits, action_val).item()

            self.val_losses.append(val_loss)

            # ログ出力
            if (epoch + 1) % 5 == 0:
                accuracy = (
                    (val_logits.argmax(dim=1) == action_val).float().mean().item()
                )
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {accuracy:.4f}"
                )

            # 早期停止
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 最良モデルを保存
                self._save_model("policy_bc_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早期停止（検証損失が改善しなくなった）")
                    break

        # 最終的に使うモデルも、最良の検証損失を記録した重みに戻す。
        best_state = torch.load("policy_bc_best.pt", map_location=self.device)
        self.model.load_state_dict(best_state)

        print(f"\n✅ 訓練完了")
        print(f"最良検証損失：{best_val_loss:.4f}")

    def _save_model(self, filepath):
        """モデルを保存"""
        torch.save(self.model.state_dict(), filepath)

    def save_model(self, filepath="policy_bc.pt"):
        """最終モデルを保存"""
        self._save_model(filepath)
        print(f"✅ モデルを {filepath} に保存しました")

    def plot_losses(self, save_path="training_loss.png"):
        """訓練曲線をプロット"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss", alpha=0.7)
        plt.plot(self.val_losses, label="Validation Loss", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"✅ 訓練曲線を {save_path} に保存しました")


if __name__ == "__main__":
    # 使用例
    trainer = BCTrainer(device="cpu")

    # デモを読み込み
    obs, actions = trainer.load_demos("demos/rule_based_demos.json")

    # 訓練
    trainer.train(obs, actions, epochs=100, batch_size=32)

    # モデルを保存
    trainer.save_model("policy_bc_final.pt")

    # 訓練曲線をプロット
    trainer.plot_losses()
