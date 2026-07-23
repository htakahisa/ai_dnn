# learning_defender.py
from collections import deque
import numpy as np
import torch
import random
from pathlib import Path
import sys

# controllers.py から親クラスをインポート
from controllers import BaseController
# 統合された最新の学習スクリプトからネットワークをインポート
from train_defender_combined import QNetwork

class LearningDefenderController(BaseController):
    """【AIモデル適用】ディフェンダーの操作クラス（ファイル分離＆BFSキャッシュ最適化版）"""
    def __init__(self, model_path="dqn_gridworld_fixedmap.pt", obs_dim=16, n_actions=4):
        super().__init__()
        # GPUが使えるならGPU、なければCPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデルの読み込み
        import sys
        from pathlib import Path

        # 一つ上の階層の「02.maru」フォルダへの絶対パスを計算
        current_dir = Path(__file__).resolve().parent
        target_dir = current_dir.parent / "02.maru"

        # 1. 検索パスに登録してインポートできるようにする
        if str(target_dir) not in sys.path:
            sys.path.append(str(target_dir))

        # 2. モデルファイル名だけが渡された場合、02.maru フォルダ内のパスに変換する
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # 相対パス（ファイル名だけなど）なら、02.maru フォルダを基準にする
            full_model_path = target_dir / model_path_obj
        else:
            full_model_path = model_path_obj

        self.model = QNetwork(obs_dim, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(str(full_model_path), map_location=self.device))
        self.model.eval()
        
        # 行動とマップのキャッシュ状態を管理（マルチエージェント対応）
        self.last_actions = {}          # 個別の行動記憶用辞書
        self.cached_planted_pos = None  # キャッシュしたスパイク位置を記録
        self.cached_dist_map = None     # キャッシュした距離マップを記録


    def _select_action_by_iq(self, q_values, char, baseline_mode="softmax"):
        """IQ100以上では従来の選択方法を維持し、100未満だけ判断ミスを加える。"""
        q_values = np.asarray(q_values, dtype=np.float64)
        valid = np.flatnonzero(np.isfinite(q_values))
        if len(valid) == 0:
            return 0

        if baseline_mode == "argmax":
            baseline_action = int(valid[np.argmax(q_values[valid])])
        else:
            valid_q = q_values[valid]
            probs = np.exp((valid_q - np.max(valid_q)) / 0.5)
            probs = probs / probs.sum()
            baseline_action = int(np.random.choice(valid, p=probs))

        iq = float(getattr(char, "effective_iq", getattr(char, "iq", 100.0)))
        decision_accuracy = max(0.0, min(1.0, iq / 100.0))

        if iq <= 100.0:
            if random.random() < decision_accuracy or len(valid) == 1:
                return baseline_action
        else:
            # 100超過分は、Softmax時に外した選択をargmaxへ戻す小さな補正にする。
            # 通常時のargmax挙動はそのままなので、既存AIを壊さない。
            best_action = int(valid[np.argmax(q_values[valid])])
            overcap_correction = min(0.20, max(0.0, (iq - 100.0) / 200.0))
            if baseline_action != best_action and random.random() < overcap_correction:
                return best_action
            return baseline_action

        alternatives = valid[valid != baseline_action]
        if len(alternatives) == 0:
            return baseline_action
        values = q_values[alternatives]
        probs = np.exp((values - np.max(values)) / 0.5)
        probs /= probs.sum()
        return int(np.random.choice(alternatives, p=probs))

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        is_planted = game_state["is_planted"]
        planted_pos = game_state["planted_pos"]
        r, c = char.pos

        # ---------------------------------------------------------------------
        # ラウンド終了時・プラント前のキャッシュリセット
        # ---------------------------------------------------------------------
        if not is_planted:
            self.cached_planted_pos = None
            self.cached_dist_map = None
            if char.name in self.last_actions:
                del self.last_actions[char.name] 
            return self.get_next_pos_random(char.pos, grid)

        # ---------------------------------------------------------------------
        # 1. プラントされた場合の挙動（作成済みAIモデルで動かす）
        # ---------------------------------------------------------------------
        if is_planted and planted_pos:
            # すでにスパイクの真上か隣接（解除可能範囲）にいる場合はその場に留まる
            dist_to_spike = max(abs(planted_pos[0] - r), abs(planted_pos[1] - c))
            if dist_to_spike <= 1:
                return char.pos

            # 現在のスパイク位置が、キャッシュした位置と違う（または未計算）なら1回だけ計算
            if self.cached_planted_pos != tuple(planted_pos):
                self.cached_planted_pos = tuple(planted_pos)
                self.cached_dist_map = self._compute_bfs_map(planted_pos, grid)

            # AIモデル用の観測データ（Observation）を作成
            obs = self._make_observation(char, planted_pos, game_state)
            
            # モデルから行動を予測
            with torch.no_grad():
                state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.model(state_t).squeeze(0)
                
                # 💡 デッドロック（上下反復ハメ）を学習能力の枠組みで回避するためSoftmaxサンプリング
                action = self._select_action_by_iq(q_values.cpu().numpy(), char, baseline_mode="softmax")

            self.last_actions[char.name] = action 
            
            # 行動番号(0~3)を座標変化に変換
            moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
            next_pos = [r + moves[action][0], c + moves[action][1]]
            
            # 移動先が壁でなければその座標を返す（壁ならその場に留まる）
            height, width = grid.shape
            if 0 <= next_pos[0] < height and 0 <= next_pos[1] < width and grid[next_pos[0], next_pos[1]] != 1:
                return next_pos
            else:
                return char.pos

    def _make_observation(self, char, target_pos, game_state):
        """環境のStateをモデルの入力（16次元）に成形する"""
        grid = game_state["grid"]
        height, width = grid.shape
        pr, pc = char.pos
        tr, tc = target_pos

        # 1. 自身とゴールの座標正規化 (4次元)
        base = [pr / (height - 1), pc / (width - 1),
                tr / (height - 1), tc / (width - 1)]

        # 2. 周囲の壁情報 (4次元) [上, 下, 左, 右]
        local_walls = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                local_walls.append(0.0)
            else:
                local_walls.append(1.0)

        # 3. 前回の行動のOne-hot表現 (4次元)
        char_last_action = self.last_actions.get(char.name, None) 
        last_action_onehot = [0.0, 0.0, 0.0, 0.0]
        if char_last_action is not None:
            last_action_onehot[char_last_action] = 1.0

        # 4. 隣接マスのゴールまでの距離（BFS）(4次元)
        max_dist = height * width
        neighbor_distances = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                d = self.cached_dist_map[nr, nc]
                neighbor_distances.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                neighbor_distances.append(1.0)

        return np.array(base + local_walls + last_action_onehot + neighbor_distances, dtype=np.float32)

    def _compute_bfs_map(self, target, grid):
        """ターゲットからの距離マップを計算する"""
        height, width = grid.shape
        dist = np.full((height, width), np.inf)
        tr, tc = target
        dist[tr, tc] = 0
        
        q = deque([(tr, tc)])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
        return dist


class LearningDefenderAllAIController(BaseController):
    """【AIモデル適用】ディフェンダー操作クラス（敵の目撃情報対応・ローテーション強化版）"""
    def __init__(self, model_path="dqn_defender_combined_best.pt", obs_dim=20, n_actions=5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        current_dir = Path(__file__).resolve().parent
        full_model_path = current_dir / model_path if not Path(model_path).is_absolute() else Path(model_path)

        # 🎯 正しいQNetworkクラスでインスタンス化
        self.model = QNetwork(obs_dim, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(str(full_model_path), map_location=self.device))
        self.model.eval()
        
        self.last_actions = {}         
        self.assigned_sites = {}       
        self.cached_target_pos = {}    
        self.cached_dist_maps = {}     
        self.pos_history = {}   # 💡追加：直近位置の履歴（デッドロック検知用）


    def _select_action_by_iq(self, q_values, char, baseline_mode="softmax"):
        """IQ100以上では従来の選択方法を維持し、100未満だけ判断ミスを加える。"""
        q_values = np.asarray(q_values, dtype=np.float64)
        valid = np.flatnonzero(np.isfinite(q_values))
        if len(valid) == 0:
            return 0

        if baseline_mode == "argmax":
            baseline_action = int(valid[np.argmax(q_values[valid])])
        else:
            valid_q = q_values[valid]
            probs = np.exp((valid_q - np.max(valid_q)) / 0.5)
            probs = probs / probs.sum()
            baseline_action = int(np.random.choice(valid, p=probs))

        iq = float(getattr(char, "effective_iq", getattr(char, "iq", 100.0)))
        decision_accuracy = max(0.0, min(1.0, iq / 100.0))

        if iq <= 100.0:
            if random.random() < decision_accuracy or len(valid) == 1:
                return baseline_action
        else:
            # 100超過分は、Softmax時に外した選択をargmaxへ戻す小さな補正にする。
            # 通常時のargmax挙動はそのままなので、既存AIを壊さない。
            best_action = int(valid[np.argmax(q_values[valid])])
            overcap_correction = min(0.20, max(0.0, (iq - 100.0) / 200.0))
            if baseline_action != best_action and random.random() < overcap_correction:
                return best_action
            return baseline_action

        alternatives = valid[valid != baseline_action]
        if len(alternatives) == 0:
            return baseline_action
        values = q_values[alternatives]
        probs = np.exp((values - np.max(values)) / 0.5)
        probs /= probs.sum()
        return int(np.random.choice(alternatives, p=probs))

    def reset_round(self):
        self.last_actions.clear()
        self.assigned_sites.clear()
        self.cached_target_pos.clear()
        self.cached_dist_maps.clear()
        self.pos_history.clear()

    def decide_move(self, char, game_state):
        grid = game_state["grid"]
        is_planted = game_state["is_planted"]
        planted_pos = game_state["planted_pos"]
        r, c = char.pos

        # 🎯 ターゲット情報の取得（初期値を学習環境と完全一致させる）
        spotted_info = game_state.get('spotted_info', {'spotted': 0.0, 'site_r': 0.0, 'site_c': 0.0})

        if is_planted and planted_pos:
            target_pos = tuple(planted_pos)
            if char.name in self.assigned_sites:
                del self.assigned_sites[char.name]

            # 💡【追加】スパイクに隣接(dist<=1)している場合は、モデルの確率的サンプリングに
            # 委ねず強制的に解除アクションを返す。移動で解除タイマーがリセットされる事故を防ぐ。
            dist_to_spike = max(abs(target_pos[0] - r), abs(target_pos[1] - c))
            if dist_to_spike <= 1:
                self.last_actions[char.name] = 4
                return char.pos, "DEFUSE"
        else:
            # 👁️ 敵が目撃されたら、その目撃されたサイトの正確な(r, c)へローテーションする
            if spotted_info.get('spotted', 0.0) > 0.5:
                target_pos = (int(spotted_info['site_r']), int(spotted_info['site_c']))
            else:
                # 目撃情報がなければ初期の割り当て防衛サイトへ
                if char.name not in self.assigned_sites:
                    plants = list(zip(*np.where(grid == 2)))
                    self.assigned_sites[char.name] = random.choice(plants) if plants else (grid.shape[0]//2, grid.shape[1]//2)
                target_pos = self.assigned_sites[char.name]

        if self.cached_target_pos.get(char.name) != target_pos:
            self.cached_target_pos[char.name] = target_pos
            self.cached_dist_maps[char.name] = self._compute_bfs_map(target_pos, grid)

        # 🧠 モデル推論
        obs = self._make_observation(char, target_pos, game_state, spotted_info)
        
        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0).cpu().numpy()

        # 💡【変更】直近数ステップで同じマスを行き来していないか確認
        history = self.pos_history.setdefault(char.name, deque(maxlen=7))
        is_stuck = len(history) == history.maxlen and len(set(map(tuple, history))) <= 2

        if is_stuck:
            # IQ実装前と同じく、詰まり時はSoftmaxで揺らす。
            action = self._select_action_by_iq(
                q_values, char, baseline_mode="softmax"
            )
        else:
            # IQ実装前と同じく、通常時はargmax。
            action = self._select_action_by_iq(
                q_values, char, baseline_mode="argmax"
            )

        self.last_actions[char.name] = action
        history.append(tuple(char.pos))

        if action == 4:
            return char.pos, "DEFUSE"

        moves = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        next_pos = [r + moves[action][0], c + moves[action][1]]
        if 0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1] and grid[next_pos[0], next_pos[1]] != 1:
            return next_pos, "MOVE"
        return char.pos, "MOVE"

    def _make_observation(self, char, target_pos, game_state, spotted_info):
        grid = game_state["grid"]
        pr, pc = char.pos
        tr, tc = target_pos
        height, width = grid.shape

        base = [pr/(height-1), pc/(width-1), tr/(height-1), tc/(width-1)]
        walls = [0.0 if self._is_walkable(pr+dr, pc+dc, grid) else 1.0 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]]
        last_act = [1.0 if self.last_actions.get(char.name) == i else 0.0 for i in range(5)]
        
        max_dist = height * width
        
        # 💡 境界外アクセスを防ぎつつ、キャッシュマップから正しく距離を取得
        dists = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pr + dr, pc + dc
            if self._is_walkable(nr, nc, grid):
                d = self.cached_dist_maps[char.name][nr, nc]
                dists.append(d / max_dist if np.isfinite(d) else 1.0)
            else:
                dists.append(1.0)
        
        # enemy_info を 3次元 (spotted, site_r, site_c) で確定
        enemy_info = [
            float(spotted_info.get('spotted', 0.0)), 
            float(spotted_info.get('site_r', 0.0)) / (height-1),
            float(spotted_info.get('site_c', 0.0)) / (width-1)
        ]

        return np.array(base + walls + last_act + dists + enemy_info, dtype=np.float32)

    def _is_walkable(self, r, c, grid):
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1

    def _compute_bfs_map(self, target, grid):
        height, width = grid.shape
        dist = np.full((height, width), np.inf)
        tr, tc = target
        dist[tr, tc] = 0
        
        q = deque([(tr, tc)])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 1:
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
        return dist