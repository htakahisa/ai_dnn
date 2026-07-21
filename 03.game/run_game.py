import random
import math
import tkinter as tk
import numpy as np

# 分離した操作用クラスをインポート
from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from learning_defender import LearningDefenderController, LearningDefenderAllAIController
from map_data import NEW_MAZE_STR
# 💡追加：試合開始前のキャラクター編成画面
from roster_select import RosterSelectScreen

WINNING_ROUNDS = 13
TICK_TIME = 1000

MAX_HP = 100
BODY_DAMAGE = 40
HEADSHOT_DAMAGE = 160
SHOOT_INTERVAL_TICKS = 1
SIDE_PANEL_WIDTH = 260
PLANT_REQUIRED_SECONDS = 4.0
DEFUSE_REQUIRED_SECONDS = 6.0
SMOKE_DURATION_SECONDS = 15.0
MOVING_ACCURACY = 0.50
MOVING_TARGET_HIT_MULTIPLIER = 0.70
BLIND_DURATION_SECONDS = 3.0
FLASH_BURST_DURATION_SECONDS = 2.0
BLIND_ACCURACY_MULTIPLIER = 0.30
FLASH_SPEED_CELLS_PER_TICK = 3
FLASH_MAX_FLIGHT_TICKS = 5
RECON_SPEED_CELLS_PER_TICK = 3
REVEAL_DURATION_SECONDS = 5.0
REVEALED_DODGE_MULTIPLIER = 0.50
RECON_REVEAL_SIZE = 9

# 設定ファイルは、このrun_game.pyと同じフォルダから読み込む。
# バージョン名や「(1)」付きファイルを直接importしないことで、
# 古いファイルを誤って読む問題を防ぐ。
import importlib.util
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent


def _load_local_module(module_name, filenames):
    """候補のうち最初に存在するローカルPythonファイルを読み込む。"""
    for filename in filenames:
        path = _BASE_DIR / filename
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"[LOAD] {module_name}: {path.name}")
            return module
        except Exception as exc:
            print(f"[LOAD ERROR] {path.name}: {exc}")
    print(f"[LOAD ERROR] {module_name}: usable file not found")
    return None


# 配布時は canonical 名（character_stats.py / player_combos.py）を使う。
_character_stats = _load_local_module(
    "game_character_stats",
    ("character_stats.py", "character_stats_dynamic.py", "character_stats_v3.py", "character_stats(1).py"),
)

_combo_module = _load_local_module(
    "game_player_combos",
    ("player_combos.py", "player_combos_v3.py", "player_combos(1).py", "player_combos_v2.py"),
)
PLAYER_COMBOS = getattr(_combo_module, "COMBOS", []) if _combo_module else []
if not isinstance(PLAYER_COMBOS, list):
    print("[LOAD ERROR] player_combos.py の COMBOS がlistではありません")
    PLAYER_COMBOS = []

_awakening_module = _load_local_module(
    "game_awakening_events",
    ("awakening_events.py",),
)
AWAKENING_EVENTS = getattr(_awakening_module, "AWAKENING_EVENTS", []) if _awakening_module else []
if not isinstance(AWAKENING_EVENTS, list):
    print("[LOAD ERROR] awakening_events.py の AWAKENING_EVENTS がlistではありません")
    AWAKENING_EVENTS = []

COMBO_DISPLAY_TICKS = 3
COMBO_BANNER_HEIGHT = 92


def _clamp_rate(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    # 50 のような百分率表記にも対応
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def calculate_combat_power(hs_rate, dodge_rate, iq, accuracy, reaction):
    """現在の5ステータスから総合戦闘力を動的に算出する。

    割合は 1.0 = 100% として扱う。
    総合戦闘力 =
        HS%*130
        + 回避率*170
        + IQ/2
        + (命中率-0.2)*130
        + 反応速度/2.2
    """
    hs_rate = _clamp_rate(hs_rate, 0.0)
    dodge_rate = _clamp_rate(dodge_rate, 0.0)
    accuracy = _clamp_rate(accuracy, 0.0)
    try:
        iq = float(iq)
    except (TypeError, ValueError):
        iq = 0.0
    try:
        reaction = float(reaction)
    except (TypeError, ValueError):
        reaction = 0.0

    return (
        hs_rate * 130.0
        + dodge_rate * 170.0
        + (iq / 2.0)
        + (accuracy - 0.2) * 130.0
        + (reaction / 2.2)
    )


def get_character_combat_stats(name):
    """キャラクター定義から命中率・弾除け率・HS率を取得する。未定義時は既定値。"""
    defaults = {
        "accuracy": 0.50,
        "dodge_rate": 0.10,
        "hs_rate": 0.20,
        "iq": 50,
        "reaction": 100,
        "role": "フラッシュ",
    }
    if _character_stats is None:
        return defaults

    raw = None
    for attr in ("CHARACTER_STATS", "character_stats", "characters", "CHARACTERS"):
        table = getattr(_character_stats, attr, None)
        if isinstance(table, dict) and name in table:
            raw = table[name]
            break

    if raw is None:
        # 現在の character_stats.py の正式な取得関数に対応
        for getter_name in ("get_by_name", "get_stats"):
            getter = getattr(_character_stats, getter_name, None)
            if callable(getter):
                try:
                    raw = getter(name)
                except Exception:
                    raw = None
                if raw is not None:
                    break

    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raw = vars(raw) if hasattr(raw, "__dict__") else {}

    def pick(keys, default):
        for key in keys:
            if key in raw:
                return _clamp_rate(raw[key], default)
        return default

    def pick_text(keys, default):
        for key in keys:
            if key in raw and raw[key] is not None:
                return str(raw[key])
        return default

    def pick_number(keys, default):
        for key in keys:
            if key in raw:
                try:
                    return float(raw[key])
                except (TypeError, ValueError):
                    pass
        return float(default)

    return {
        "accuracy": pick(("accuracy", "aim", "hit_rate", "hit_pct", "命中率"), defaults["accuracy"]),
        "dodge_rate": pick(("dodge_rate", "dodge", "dodge_pct", "evasion", "弾除け率"), defaults["dodge_rate"]),
        "hs_rate": pick(("hs_rate", "hs", "hs_pct", "headshot_rate", "HS", "HS%"), defaults["hs_rate"]),
        "iq": pick_number(("iq", "IQ", "判断力"), defaults["iq"]),
        "reaction": pick_number(("reaction", "reaction_speed", "反応速度"), defaults["reaction"]),
        "role": pick_text(("role", "ロール"), defaults["role"]),
    }


def get_all_character_names():
    """character_stats.py に登録された全キャラクター名を取得する。"""
    if _character_stats is None:
        return []

    all_names = getattr(_character_stats, "all_names", None)
    if callable(all_names):
        try:
            names = all_names()
            if names:
                return [str(name) for name in names]
        except Exception:
            pass

    all_characters = getattr(_character_stats, "all_characters", None)
    if callable(all_characters):
        try:
            result = []
            for char in all_characters():
                if isinstance(char, dict):
                    name = char.get("name")
                else:
                    name = getattr(char, "name", None)
                if name:
                    result.append(str(name))
            if result:
                return result
        except Exception:
            pass

    for attr in ("CHARACTER_TABLE", "CHARACTER_STATS", "character_stats", "characters", "CHARACTERS"):
        table = getattr(_character_stats, attr, None)
        if isinstance(table, dict):
            return [str(name) for name in table.keys()]
        if isinstance(table, (list, tuple)):
            result = []
            for char in table:
                if isinstance(char, dict):
                    name = char.get("name")
                else:
                    name = getattr(char, "name", None)
                if name:
                    result.append(str(name))
            if result:
                return result
    return []


def _print_data_diagnostics():
    names = get_all_character_names()
    print(f"[DATA] characters={len(names)} / combos={len(PLAYER_COMBOS)} / awakenings={len(AWAKENING_EVENTS)}")
    if not names:
        print("[DATA ERROR] キャラクター一覧が0件です。character_stats.pyをrun_game.pyと同じフォルダに置いてください。")
    if not PLAYER_COMBOS:
        print("[DATA WARNING] コンボが0件です。player_combos.pyを確認してください。")


_print_data_diagnostics()


class Character:
    def __init__(self, name, team, pos, text_color, bg_color, has_spike=False, kills=0, deaths=0):
        self.name = name
        self.base_name = name
        self.display_name = name
        self.team = team
        self.pos = list(pos)
        self.text_color = text_color
        self.bg_color = bg_color
        self.is_alive = True
        self.just_died = False
        self.has_spike = has_spike
        self.plant_timer = 0
        self.is_planting = False
        self.defuse_timer = 0
        self.max_hp = MAX_HP
        self.hp = MAX_HP
        self.kills = kills
        self.deaths = deaths
        # 覚醒条件用。試合通算キルとは別に、各ラウンド開始時に0から数える。
        self.round_kills = 0

        stats = get_character_combat_stats(name)
        self.role = stats.get("role", "フラッシュ")
        self.accuracy = stats["accuracy"]
        self.dodge_rate = stats["dodge_rate"]
        self.hs_rate = stats["hs_rate"]
        self.iq = stats.get("iq", 50.0)
        self.reaction = stats.get("reaction", 100.0)
        # タイガーの固有パッシブ「ハンター」：常時 Hit% を10ポイント、HS% を5ポイント上昇。
        self.hunter_active = self.role == "タイガー"
        if self.hunter_active:
            self.accuracy = min(1.0, self.accuracy + 0.10)
            self.hs_rate = min(1.0, self.hs_rate + 0.05)
        self.ability_name = {
            "フラッシュ": "FLASH",
            "スモーカー": "SMOKE",
            "シーカー": "RECON",
            "タイガー": "HUNT",
        }.get(self.role, "FLASH")
        self.moved_this_tick = False
        self.smoke_charges = 1 if self.ability_name == "SMOKE" else 0
        self.flash_charges = 1 if self.ability_name == "FLASH" else 0
        self.recon_charges = 1 if self.ability_name == "RECON" else 0
        self.blind_remaining = 0.0
        self.reveal_remaining = 0.0
        self.los_revealed = False
        # このラウンドで発動しているプレイヤーコンボ名。
        self.active_combos = []
        self.active_awakening = None
        self.triggered_awakening_events = set()

    @property
    def combat_power(self):
        """補正後の現在ステータスから総合戦闘力を毎回計算する。"""
        return calculate_combat_power(
            self.hs_rate, self.dodge_rate, self.iq, self.accuracy, self.reaction
        )

def _canonical_combo_stat_key(key):
    """コンボ定義内の表記ゆれを Character の属性名へ変換する。"""
    normalized = str(key).strip().lower().replace("％", "%").replace(" ", "")
    aliases = {
        "accuracy": "accuracy",
        "hit": "accuracy",
        "hit%": "accuracy",
        "hit_pct": "accuracy",
        "hit_rate": "accuracy",
        "命中率": "accuracy",
        "hs": "hs_rate",
        "hs%": "hs_rate",
        "hs_pct": "hs_rate",
        "hs_rate": "hs_rate",
        "headshot_rate": "hs_rate",
        "ヘッドショット率": "hs_rate",
        "dodge": "dodge_rate",
        "dodge%": "dodge_rate",
        "dodge_pct": "dodge_rate",
        "dodge_rate": "dodge_rate",
        "回避率": "dodge_rate",
        "弾除け率": "dodge_rate",
        "reaction": "reaction",
        "reaction_speed": "reaction",
        "反応速度": "reaction",
        "hp": "max_hp",
        "max_hp": "max_hp",
    }
    return aliases.get(normalized)


def _apply_combo_bonus(character, stat_key, value):
    """一つのコンボ補正を適用する。率は加算後0～100%に収める。"""
    attr = _canonical_combo_stat_key(stat_key)
    if attr is None:
        return False
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return False

    if attr in ("accuracy", "hs_rate", "dodge_rate"):
        # 10 のような指定も10ポイントとして扱う。
        if abs(amount) > 1.0:
            amount /= 100.0
        setattr(character, attr, max(0.0, min(1.0, getattr(character, attr) + amount)))
    elif attr == "reaction":
        character.reaction = max(0.0, character.reaction + amount)
    elif attr == "max_hp":
        old_max = character.max_hp
        character.max_hp = max(1, int(round(character.max_hp + amount)))
        character.hp = max(0, min(character.max_hp, character.hp + (character.max_hp - old_max)))
    return True


class VisualFPSBattle:
    def __init__(self, maze_str, attacker_controller, defender_controller, headless=False, attacker_roster=None, spike_holder_name=None):
        self.maze_str = maze_str
        self.headless = headless # 【新機能】画面を描画しない設定
        # 💡編成画面で選択されたアタッカー5人の名前リスト（Noneなら Att1, Att2... のデフォルト名になる）
        self.attacker_roster = attacker_roster
        self.spike_holder_name = spike_holder_name
        self.defender_roster = None  # 試合開始時に一度だけ決定し、全ラウンドで固定
        
        lines = [line.strip() for line in maze_str.strip("\n").split("\n") if line.strip()]
        self.height, self.width = len(lines), len(lines[0])
        self.grid = np.array([[int(ch) for ch in line] for line in lines], dtype=np.int32)
        self.cell_size = 24
        
        self.attacker_controller = attacker_controller
        self.defender_controller = defender_controller
        
        self.attacker_wins = 0
        self.defender_wins = 0
        self.current_round = 1
        self.battle_tick = 0
        self.match_stats = {}  # 名前 -> {kills, deaths}。ラウンドをまたいで保持
        self.map_offset_x = SIDE_PANEL_WIDTH
        self.map_pixel_width = self.width * self.cell_size
        self.map_pixel_height = self.height * self.cell_size
        self.ability_area_height = 110
        
        # 画面非表示（headless）モードの時はTkinterを立ち上げない
        if not self.headless:
            self.root = tk.Tk()
            self.root.title("Attacker vs Defender")
            self.canvas = tk.Canvas(
                self.root,
                width=self.map_pixel_width + SIDE_PANEL_WIDTH * 2,
                height=self.map_pixel_height + self.ability_area_height + COMBO_BANNER_HEIGHT,
                bg="#10141c",
                highlightthickness=0,
            )
            self.canvas.pack(fill="both", expand=True)
            self.root.minsize(self.map_pixel_width + SIDE_PANEL_WIDTH * 2, self.map_pixel_height + self.ability_area_height + COMBO_BANNER_HEIGHT + 30)
            self.canvas.bind("<Button-1>", self.on_canvas_click)   # 💡追加
            self.label = tk.Label(self.root, text="Round 1 Start", font=("Arial", 10))
            self.label.pack()
        
        self.match_over = False
        self.init_round()

    def init_round(self):
        self.round_over = False
        
        area_3 = list(zip(*np.where(self.grid == 3)))
        area_4 = list(zip(*np.where(self.grid == 4)))
        
        spike_holder_index = random.randint(0, len(area_3) - 1) if area_3 else -1
        if self.attacker_roster and self.spike_holder_name in self.attacker_roster:
            spike_holder_index = self.attacker_roster.index(self.spike_holder_name)
        
        self.chars = []
        for i, pos in enumerate(area_3):
            has_spike = (i == spike_holder_index)
            # 💡編成画面でロスターが選ばれていればその名前を使い、なければデフォルト名(Att1, Att2...)
            if self.attacker_roster and i < len(self.attacker_roster):
                name = self.attacker_roster[i]
            else:
                name = f"Att{i+1}"
            saved = self.match_stats.setdefault(name, {"kills": 0, "deaths": 0})
            self.chars.append(Character(name, "A", pos, "white", "#c0392b", has_spike=has_spike,
                                        kills=saved["kills"], deaths=saved["deaths"]))
        # ディフェンダー編成は試合開始時に一度だけランダム決定し、以後のラウンドで固定する。
        if self.defender_roster is None:
            registered_names = get_all_character_names()
            attacker_names = set(self.attacker_roster or [])
            defender_pool = [name for name in registered_names if name not in attacker_names]
            if len(defender_pool) < len(area_4):
                defender_pool = list(registered_names)

            if defender_pool:
                if len(defender_pool) >= len(area_4):
                    self.defender_roster = random.sample(defender_pool, len(area_4))
                else:
                    self.defender_roster = [random.choice(defender_pool) for _ in area_4]
            else:
                self.defender_roster = [f"Def{i+1}" for i in range(len(area_4))]

        defender_names = self.defender_roster

        for i, pos in enumerate(area_4):
            name = defender_names[i]
            saved = self.match_stats.setdefault(name, {"kills": 0, "deaths": 0})
            self.chars.append(Character(name, "D", pos, "white", "#27ae60",
                                        kills=saved["kills"], deaths=saved["deaths"]))

        # ロール補正を含む基礎能力の初期化後、チーム編成によるコンボ補正を適用する。
        # Character はラウンドごとに作り直されるため、補正が累積することはない。
        self._apply_player_combos()
        # コンボと覚醒イベントを同じ告知キューで順番に表示する。
        self.announcement_queue = []
        self.combo_announcement_index = 0
        self.combo_announcement_ticks_left = 0
        for combo_announcement in self.active_player_combos:
            item = dict(combo_announcement)
            item.setdefault("type", "combo")
            self._enqueue_announcement(item)
            
        plants = list(zip(*np.where(self.grid == 2)))
        self.target_plant_pos = random.choice(plants) if plants else None
            
        self.spike_pos = None          
        self.is_planted = False        
        self.planted_pos = None        
        self.round_timer = 90          
        self.detonate_timer = 45       
        self.is_defused = False        
        self.active_defuser_name = None
        self.last_engagements = []
        self.last_shot = None
        self.last_shots = []
        self.battle_tick = 0
        self.smokes = []  # {cells:set[(r,c)], remaining_seconds:float, owner:str}
        self.flash_projectiles = []  # {owner, team, path, progress, ticks_alive}
        self.recon_projectiles = []  # {owner, team, path, progress}
        self.flash_bursts = []  # {pos, remaining_seconds}
        self.recon_bursts = []  # {cells, remaining_seconds}
        self.ability_mode = None
        
        # ディフェンダーコントローラの内部状態(サイト割り当て等)をリセットする
        if hasattr(self.defender_controller, "reset_round"):
            self.defender_controller.reset_round()
            
        # 💡追加：アタッカー側も同様にリセット(UserInputController用)
        if hasattr(self.attacker_controller, "reset_round"):
            self.attacker_controller.reset_round()

    def _apply_player_combos(self):
        """同じチームに必要メンバーが全員いるコンボを発動する。"""
        self.active_player_combos = []
        for team in ("A", "D"):
            team_chars = [char for char in self.chars if char.team == team]
            chars_by_name = {char.name: char for char in team_chars}
            team_names = set(chars_by_name)

            for combo in PLAYER_COMBOS:
                if not isinstance(combo, dict):
                    continue
                combo_name = str(combo.get("name", "名称未設定コンボ"))
                required_players = tuple(str(name) for name in combo.get("players", ()))
                if not required_players or not set(required_players).issubset(team_names):
                    continue

                common_bonuses = combo.get("bonuses", {})
                per_player_bonuses = combo.get("player_bonuses", {})
                renames = combo.get("renames", {})
                affected = []

                for player_name in required_players:
                    char = chars_by_name[player_name]
                    if isinstance(common_bonuses, dict):
                        for stat_key, value in common_bonuses.items():
                            _apply_combo_bonus(char, stat_key, value)
                    own_bonuses = per_player_bonuses.get(player_name, {}) if isinstance(per_player_bonuses, dict) else {}
                    if isinstance(own_bonuses, dict):
                        for stat_key, value in own_bonuses.items():
                            _apply_combo_bonus(char, stat_key, value)
                    if isinstance(renames, dict) and player_name in renames:
                        char.display_name = str(renames[player_name])
                    if combo_name not in char.active_combos:
                        char.active_combos.append(combo_name)
                    affected.append(player_name)

                self.active_player_combos.append({
                    "type": "combo",
                    "name": combo_name,
                    "team": team,
                    "players": tuple(affected),
                    "display_players": tuple(chars_by_name[n].display_name for n in affected),
                    "effect_text": str(combo.get("effect_text") or self._describe_bonuses(common_bonuses, per_player_bonuses)),
                })

    def _describe_bonuses(self, common_bonuses, per_player_bonuses=None):
        labels = {"accuracy": "Hit%", "hs_rate": "HS%", "dodge_rate": "回避率", "reaction": "反応速度", "max_hp": "最大HP"}
        parts = []
        if isinstance(common_bonuses, dict):
            for key, value in common_bonuses.items():
                canonical = _canonical_combo_stat_key(key)
                if canonical:
                    amount = float(value)
                    shown = amount * 100 if canonical != "max_hp" and abs(amount) <= 1 else amount
                    parts.append(f"{labels.get(canonical, canonical)} {shown:+g}")
        if isinstance(per_player_bonuses, dict) and per_player_bonuses:
            parts.append("個別補正あり")
        return " / ".join(parts) if parts else "特殊効果"

    def _enqueue_announcement(self, announcement):
        """コンボ・覚醒イベントの告知を表示待ちキューへ追加する。"""
        if not isinstance(announcement, dict):
            return
        if not hasattr(self, "announcement_queue"):
            self.announcement_queue = []
        self.announcement_queue.append(announcement)

        # 現在何も表示していない場合は、追加された告知を直ちに表示開始する。
        if self.combo_announcement_index >= len(self.announcement_queue) - 1:
            self.combo_announcement_index = len(self.announcement_queue) - 1
            self.combo_announcement_ticks_left = COMBO_DISPLAY_TICKS

    def _advance_combo_announcement(self):
        """現在の告知を1Tick進め、終了したら次の告知へ移る。"""
        if not getattr(self, "announcement_queue", None):
            return
        if self.combo_announcement_index >= len(self.announcement_queue):
            return
        self.combo_announcement_ticks_left -= 1
        if self.combo_announcement_ticks_left <= 0:
            self.combo_announcement_index += 1
            if self.combo_announcement_index < len(self.announcement_queue):
                self.combo_announcement_ticks_left = COMBO_DISPLAY_TICKS

    def _get_character_preset(self, name):
        if _character_stats is None:
            return None
        getter = getattr(_character_stats, "get_by_name", None)
        raw = getter(name) if callable(getter) else None
        if raw is None:
            table = getattr(_character_stats, "CHARACTER_TABLE", {})
            raw = table.get(name) if isinstance(table, dict) else None
        return raw

    def _apply_awakening_preset(self, char, preset_name):
        raw = self._get_character_preset(preset_name)
        if raw is None:
            return
        data = vars(raw) if hasattr(raw, "__dict__") else raw
        char.accuracy = _clamp_rate(data.get("hit_pct", data.get("accuracy")), char.accuracy)
        char.hs_rate = _clamp_rate(data.get("hs_pct", data.get("hs_rate")), char.hs_rate)
        char.dodge_rate = _clamp_rate(data.get("dodge_pct", data.get("dodge_rate")), char.dodge_rate)
        try:
            char.iq = float(data.get("iq", data.get("IQ", char.iq)))
        except (TypeError, ValueError):
            pass
        try:
            char.reaction = float(
                data.get("reaction", data.get("reaction_speed", data.get("反応速度", char.reaction)))
            )
        except (TypeError, ValueError):
            pass
        role = str(data.get("role", char.role))
        char.role = role
        char.ability_name = {"フラッシュ":"FLASH", "スモーカー":"SMOKE", "シーカー":"RECON", "タイガー":"HUNT"}.get(role, char.ability_name)
        char.hunter_active = role == "タイガー"
        char.smoke_charges = 1 if char.ability_name == "SMOKE" else 0
        char.flash_charges = 1 if char.ability_name == "FLASH" else 0
        char.recon_charges = 1 if char.ability_name == "RECON" else 0

    def _awakening_condition_met(self, event, char):
        condition = str(event.get("condition", ""))
        value = event.get("condition_value")
        if condition == "all_allies_dead":
            allies = [c for c in self.chars if c.team == char.team and c is not char]
            return char.is_alive and bool(allies) and all(not c.is_alive for c in allies)
        if condition == "hp_at_or_below":
            try: return char.is_alive and char.hp <= float(value)
            except (TypeError, ValueError): return False
        if condition == "kills_at_least":
            try:
                return char.is_alive and char.round_kills >= int(value)
            except (TypeError, ValueError):
                return False
        return False

    def _check_awakening_events(self):
        for event in AWAKENING_EVENTS:
            if not isinstance(event, dict):
                continue
            player = str(event.get("player", ""))
            event_name = str(event.get("name", "名称未設定の覚醒"))
            char = next((c for c in self.chars if c.base_name == player), None)
            if char is None or event_name in char.triggered_awakening_events:
                continue
            if not self._awakening_condition_met(event, char):
                continue
            preset = event.get("transform_to")
            if preset:
                self._apply_awakening_preset(char, str(preset))
            bonuses = event.get("bonuses", {})
            if isinstance(bonuses, dict):
                for key, value in bonuses.items():
                    _apply_combo_bonus(char, key, value)
            if event.get("rename"):
                char.display_name = str(event["rename"])
            char.active_awakening = event_name
            char.triggered_awakening_events.add(event_name)

            # 覚醒した瞬間に、プレイヤーコンボと同じ上部パネルへ表示する。
            # 同一Tickに複数人が覚醒した場合も、追加された順に3Tickずつ表示される。
            effect_text = str(event.get("effect_text") or self._describe_bonuses(bonuses))
            self._enqueue_announcement({
                "type": "awakening",
                "name": event_name,
                "team": char.team,
                "players": (char.base_name,),
                "display_players": (char.display_name,),
                "effect_text": effect_text,
            })

    def _tick_seconds(self):
        """1シミュレーションTickが表す秒数。時間制効果はすべてこれを使う。"""
        return max(0.001, TICK_TIME / 1000.0)

    def move_character(self, char):
        r, c = char.pos
        old_pos = tuple(char.pos)
        char.moved_this_tick = False
        
        # ---------------------------------------------------------------------
        # プラント処理
        # ユーザー操作時は、プラントゾーンに入っただけでは開始しない。
        # 選択中キャラクターの下部UIに出るPLANTボタンから明示的に開始する。
        # AI操作時のみ、従来どおり目標サイト到達時に自動で開始する。
        # ---------------------------------------------------------------------
        if char.team == "A" and char.has_spike and not self.is_planted:
            is_user_controlled = isinstance(self.attacker_controller, UserInputController)
            on_plant_site = (self.grid[r, c] == 2) if is_user_controlled else (
                self.target_plant_pos and list(char.pos) == list(self.target_plant_pos)
            )
            should_plant = char.is_planting if is_user_controlled else bool(on_plant_site)

            if should_plant and on_plant_site:
                char.plant_timer += self._tick_seconds()
                if char.plant_timer >= PLANT_REQUIRED_SECONDS:
                    self.is_planted = True
                    self.planted_pos = (r, c)
                    char.has_spike = False
                    char.plant_timer = 0
                    char.is_planting = False
                return  # 設置中は移動・射撃を行わない
            if char.is_planting and not on_plant_site:
                char.is_planting = False
            char.plant_timer = 0

        # ---------------------------------------------------------------------
        # AIにどう動くか（または解除するか）を聞く
        # ---------------------------------------------------------------------
        # 💡 プラント状態に応じた適切なターゲット座標の確定
        if self.is_planted:
            site_r = float(self.planted_pos[0]) if self.planted_pos else 0.0
            site_c = float(self.planted_pos[1]) if self.planted_pos else 0.0
        else:
            site_r = float(self.target_plant_pos[0]) if self.target_plant_pos else 0.0
            site_c = float(self.target_plant_pos[1]) if self.target_plant_pos else 0.0

        game_state = {
            "grid": self.grid,
            "spike_pos": self.spike_pos,
            "is_planted": self.is_planted,
            "planted_pos": self.planted_pos,
            "target_plant_pos": self.target_plant_pos,
            "chars": self.chars,
            "spotted_info": self.get_spotted_info() if not self.is_planted else {
                'spotted': 1.0,
                'site_r': site_r,
                'site_c': site_c
            }
        }

        if char.team == "A":
            # アタッカー側（位置だけ返す想定）
            next_pos = self.attacker_controller.decide_move(char, game_state)
            action_type = "MOVE"
        else:
            # ディフェンダー側：コントローラーによって戻り値の数が異なるため自動判別
            result = self.defender_controller.decide_move(char, game_state)
            if isinstance(result, tuple) and len(result) == 2:
                # LearningDefenderAllAIController などのアクションタイプ付きの戻り値
                next_pos, action_type = result
            else:
                # DefaultDefenderController などの座標のみの戻り値
                next_pos = result
                action_type = "MOVE"

        # ---------------------------------------------------------------------
        #  アクションタイプに応じたシステム処理 (修正版)
        # ---------------------------------------------------------------------
        if action_type == "DEFUSE":
            if self.is_planted and self.planted_pos and char.team == "D":
                dist = max(abs(self.planted_pos[0] - r), abs(self.planted_pos[1] - c))
                if dist <= 1:
                    # 同時に解除できるのは一人だけ。担当者がいない時だけロックを取得する。
                    if self.active_defuser_name in (None, char.name):
                        self.active_defuser_name = char.name
                        char.defuse_timer += self._tick_seconds()
                        # 解除完了は射撃解決後に判定する。最終解除tickでも射撃を先に解決する。
                        return
                    # 別のキャラクターが解除中なら、このキャラクターは解除を開始できない。
                    char.defuse_timer = 0
                    return
            if self.active_defuser_name == char.name:
                self.active_defuser_name = None
            char.defuse_timer = 0

        else:
            # MOVE アクションの処理。解除担当者が解除をやめたらロックを解放する。
            if self.active_defuser_name == char.name:
                self.active_defuser_name = None
            char.defuse_timer = 0  
            # 💡 インデックス参照エラーを防ぐため、next_pos が有効な2次元座標であることを保証
            if isinstance(next_pos, (list, np.ndarray)) and len(next_pos) == 2:
                nr, nc = int(next_pos[0]), int(next_pos[1])
                in_bounds = 0 <= nr < self.height and 0 <= nc < self.width
                occupied = any(
                    other is not char and other.is_alive and tuple(other.pos) == (nr, nc)
                    for other in self.chars
                )
                if in_bounds and self.grid[nr, nc] != 1 and not occupied:
                    char.pos = [nr, nc]

        char.moved_this_tick = tuple(char.pos) != old_pos

    def _smoke_cells(self):
        cells = set()
        for smoke in self.smokes:
            cells.update(smoke["cells"])
        return cells

    def _line_cells(self, start, end):
        """2マス間を結ぶBresenham線上のセルを順番に返す。"""
        y0, x0 = int(start[0]), int(start[1])
        y1, x1 = int(end[0]), int(end[1])
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        cells = []
        while True:
            cells.append((y0, x0))
            if x0 == x1 and y0 == y1:
                return cells
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _smoke_allows_line(self, line_cells, smoke_cells):
        """指定された射線がスモーク規則上通るか判定する。

        通る:
        - 始点と終点がどちらもスモーク内

        通らない:
        - 外から中
        - 中から外
        - 外から外で、途中にスモークがある
        """
        if not line_cells:
            return True

        start_in_smoke = line_cells[0] in smoke_cells
        end_in_smoke = line_cells[-1] in smoke_cells

        if start_in_smoke and end_in_smoke:
            return True

        if start_in_smoke != end_in_smoke:
            return False

        return not any(cell in smoke_cells for cell in line_cells[1:-1])

    def check_cell_line_of_sight(self, start, end, block_smoke=True):
        line_cells = self._line_cells(start, end)

        for r, c in line_cells:
            if self.grid[r, c] == 1:
                return False

        if block_smoke and not self._smoke_allows_line(line_cells, self._smoke_cells()):
            return False

        return True

    def check_line_of_sight(self, p1, p2):
        """壁とスモーク規則を考慮して、2人の間に射線が通るか判定する。"""
        line_cells = self._line_cells(tuple(p1.pos), tuple(p2.pos))

        for r, c in line_cells:
            if self.grid[r, c] == 1:
                return False

        return self._smoke_allows_line(line_cells, self._smoke_cells())

    def get_viewer_team(self):
        controllers = self.get_user_controllers()
        return controllers[0][1] if controllers else None

    def is_visible_to_team(self, target, viewer_team):
        if viewer_team is None or target.team == viewer_team:
            return True
        if target.reveal_remaining > 0:
            return True
        return any(ally.is_alive and ally.team == viewer_team and self.check_line_of_sight(ally, target)
                   for ally in self.chars)

    def _update_los_reveal(self):
        """敵同士の射線が通っている間、双方をリビール状態として扱う。"""
        for char in self.chars:
            char.los_revealed = False
        alive = [char for char in self.chars if char.is_alive]
        for i, first in enumerate(alive):
            for second in alive[i + 1:]:
                if first.team != second.team and self.check_line_of_sight(first, second):
                    first.los_revealed = True
                    second.los_revealed = True

    def _is_revealed(self, char):
        return char.reveal_remaining > 0 or char.los_revealed

    def _current_los_revealed_names(self):
        """現在、敵との射線が通っているキャラクター名の集合を返す。"""
        revealed = set()
        alive = [char for char in self.chars if char.is_alive]
        for i, first in enumerate(alive):
            for second in alive[i + 1:]:
                if first.team != second.team and self.check_line_of_sight(first, second):
                    revealed.add(first.name)
                    revealed.add(second.name)
        return revealed

    def _is_revealed_for_shot(self, char, current_los_revealed_names):
        """射撃判定用のリビール状態。

        リコン由来のリビールは即時適用する。
        射線由来は「前Tickでもリビール済み」かつ「現在も射線が通る」場合だけ
        回避率低下を適用する。これにより初めて視認したTickは通常回避率で撃たれ、
        その射撃後から射線リビール状態になる。
        """
        recon_revealed = char.reveal_remaining > 0
        persistent_los_revealed = char.los_revealed and char.name in current_los_revealed_names
        return recon_revealed or persistent_los_revealed

    def get_user_controllers(self):
        """ユーザー操作のコントローラーとそのチームのペアを返す"""
        result = []
        if isinstance(self.attacker_controller, UserInputController):
            result.append((self.attacker_controller, "A"))
        if isinstance(self.defender_controller, UserInputController):
            result.append((self.defender_controller, "D"))
        return result

    def on_canvas_click(self, event):
        if event.y < COMBO_BANNER_HEIGHT:
            return
        event.y -= COMBO_BANNER_HEIGHT
        selected = self._selected_user_character()

        # プラント可能な選択キャラクターにだけ表示されるPLANTボタン。
        plant_bounds = self._plant_button_bounds()
        if selected and plant_bounds and plant_bounds[0] <= event.x <= plant_bounds[2] and plant_bounds[1] <= event.y <= plant_bounds[3]:
            selected.is_planting = not selected.is_planting
            selected.plant_timer = 0
            self.ability_mode = None
            self.draw()
            return

        # 選択中キャラクターは、ロールに対応したアビリティ一つだけ使用できる。
        if selected and selected.ability_name in ("SMOKE", "FLASH", "RECON"):
            ability_name = selected.ability_name
            panel = self._ability_button_bounds(ability_name)
            if panel and panel[0] <= event.x <= panel[2] and panel[1] <= event.y <= panel[3]:
                same_is_armed = self.ability_mode == (ability_name, selected.team, selected.name)
                if same_is_armed:
                    self.ability_mode = None
                else:
                    charges = {
                        "SMOKE": selected.smoke_charges,
                        "FLASH": selected.flash_charges,
                        "RECON": selected.recon_charges,
                    }[ability_name]
                    if charges > 0:
                        self.ability_mode = (ability_name, selected.team, selected.name)
                self.draw()
                return

        map_x = event.x - self.map_offset_x
        if map_x < 0 or map_x >= self.map_pixel_width:
            return
        c = map_x // self.cell_size
        r = event.y // self.cell_size
        if not (0 <= r < self.height and 0 <= c < self.width):
            return

        if self.ability_mode:
            ability_name, team, owner_name = self.ability_mode
            owner = next((ch for ch in self.chars if ch.name == owner_name and ch.is_alive), None)
            if owner and owner.team == team and owner.ability_name == ability_name and self.grid[r, c] != 1:
                if ability_name == "SMOKE" and owner.smoke_charges > 0:
                    cells = {(rr, cc) for rr in range(r-1, r+2) for cc in range(c-1, c+2)
                             if 0 <= rr < self.height and 0 <= cc < self.width and self.grid[rr, cc] != 1}
                    self.smokes.append({"cells": cells, "remaining_seconds": SMOKE_DURATION_SECONDS, "owner": owner.name})
                    owner.smoke_charges -= 1
                elif ability_name == "FLASH" and owner.flash_charges > 0:
                    path = self._projectile_path(tuple(owner.pos), (r, c))
                    if len(path) > 1:
                        self.flash_projectiles.append({
                            "owner": owner.name, "team": owner.team, "path": path,
                            "progress": 0, "ticks_alive": 0
                        })
                        owner.flash_charges -= 1
                elif ability_name == "RECON" and owner.recon_charges > 0:
                    path = self._projectile_path(tuple(owner.pos), (r, c))
                    if len(path) > 1:
                        self.recon_projectiles.append({
                            "owner": owner.name, "team": owner.team, "path": path, "progress": 0
                        })
                        owner.recon_charges -= 1
            self.ability_mode = None
            self.draw()
            return

        for ctrl, team in self.get_user_controllers():
            ctrl.handle_click(r, c, self.grid, self.chars, team)
        self.draw()

    def _selected_user_character(self):
        for ctrl, team in self.get_user_controllers():
            if ctrl.selected_char:
                return next((ch for ch in self.chars if ch.name == ctrl.selected_char and ch.team == team and ch.is_alive), None)
        return None

    def _can_selected_plant(self):
        selected = self._selected_user_character()
        if not selected or not selected.is_alive or selected.team != "A":
            return False
        if self.is_planted or not selected.has_spike:
            return False
        r, c = selected.pos
        return self.grid[r, c] == 2

    def _bottom_control_layout(self):
        total_w, h = min(690, self.map_pixel_width - 20), 68
        x1 = self.map_offset_x + (self.map_pixel_width - total_w) / 2
        y1 = self.map_pixel_height + 28
        if self._can_selected_plant():
            gap, plant_w = 10, 150
            return (x1, y1, x1 + total_w - plant_w - gap, y1 + h), (x1 + total_w - plant_w, y1, x1 + total_w, y1 + h)
        return (x1, y1, x1 + total_w, y1 + h), None

    def _ability_panel_bounds(self):
        if not self._selected_user_character():
            return None
        return self._bottom_control_layout()[0]

    def _plant_button_bounds(self):
        if not self._can_selected_plant():
            return None
        return self._bottom_control_layout()[1]

    def _ability_button_bounds(self, ability_name):
        panel = self._ability_panel_bounds()
        selected = self._selected_user_character()
        if not panel or not selected or selected.ability_name != ability_name:
            return None
        return panel

    def _projectile_path(self, start, aimed_cell):
        """指定マスを方向として、壁またはマップ端まで伸びる投射経路を作る。"""
        sr, sc = start
        ar, ac = aimed_cell
        dr, dc = ar - sr, ac - sc
        if dr == 0 and dc == 0:
            return [start]
        scale = max(self.height, self.width) * 3
        far = (sr + dr * scale, sc + dc * scale)
        raw = self._line_cells(start, far)
        path = [start]
        for rr, cc in raw[1:]:
            if not (0 <= rr < self.height and 0 <= cc < self.width):
                break
            if self.grid[rr, cc] == 1:
                break
            path.append((rr, cc))
        return path

    def get_spotted_info(self):
        spike_holder = next((c for c in self.chars if c.is_alive and c.team == "A" and c.has_spike), None)
        if spike_holder is None:
            return {'spotted': 0.0, 'site_r': 0.0, 'site_c': 0.0}

        for d in self.chars:
            if d.is_alive and d.team == "D" and self.check_line_of_sight(d, spike_holder):
                return {'spotted': 1.0, 'site_r': float(spike_holder.pos[0]), 'site_c': float(spike_holder.pos[1])}

        return {'spotted': 0.0, 'site_r': 0.0, 'site_c': 0.0}


    def check_match_winner(self):
        if self.attacker_wins >= WINNING_ROUNDS:
            if not self.headless:
                self.label.config(text=f"🏆 MATCH OVER: Attacker WINS! ({self.attacker_wins} - {self.defender_wins})", fg="#c0392b", font=("Arial", 12, "bold"))
            print(f"MATCH OVER: Attacker WINS! ({self.attacker_wins} - {self.defender_wins})")
            self.match_over = True
        elif self.defender_wins >= WINNING_ROUNDS:
            if not self.headless:
                self.label.config(text=f"🏆 MATCH OVER: Defender WINS! ({self.defender_wins} - {self.attacker_wins})", fg="#27ae60", font=("Arial", 12, "bold"))
            print(f"MATCH OVER: Defender WINS! ({self.defender_wins} - {self.attacker_wins})")
            self.match_over = True
        else:
            self.current_round += 1
            if not self.headless:
                # プレイ用の時は2秒ディレイをかける
                self.root.after(2000, self.init_next_round_delayed)
            else:
                # 学習用の時はディレイなしで即時次ラウンドへ
                self.init_round()

    def init_next_round_delayed(self):
        self.init_round()
        self.loop()

    def _kill_character(self, shooter, target):
        target.hp = 0
        target.is_alive = False
        target.just_died = True
        target.deaths += 1
        shooter.kills += 1
        shooter.round_kills += 1
        self.match_stats.setdefault(target.name, {"kills": 0, "deaths": 0})["deaths"] = target.deaths
        self.match_stats.setdefault(shooter.name, {"kills": 0, "deaths": 0})["kills"] = shooter.kills
        target.is_planting = False
        target.plant_timer = 0
        if self.active_defuser_name == target.name:
            self.active_defuser_name = None
        if target.has_spike:
            self.spike_pos = tuple(target.pos)
            target.has_spike = False

    def _resolve_all_shots(self, engagements=None, current_los_revealed_names=None):
        """同Tickの射撃を反応速度が高い順に逐次処理する。

        Tick開始時に射撃予定者と標的を確定する。
        反応速度の高い射手から順に射撃し、同値の場合だけランダム順にする。
        自分の射撃順が来る前に死亡した射手は射撃できない。
        """
        if self.battle_tick % SHOOT_INTERVAL_TICKS != 0:
            self.last_shots = []
            self.last_shot = None
            return

        alive_at_tick_start = [c for c in self.chars if c.is_alive]
        shot_intents = []
        executed_shots = []

        if current_los_revealed_names is None:
            current_los_revealed_names = self._current_los_revealed_names()

        for shooter in alive_at_tick_start:
            if shooter.plant_timer > 0 or shooter.defuse_timer > 0:
                continue

            possible_targets = [
                target for target in alive_at_tick_start
                if target.team != shooter.team
                and self.check_line_of_sight(shooter, target)
            ]
            if not possible_targets:
                continue

            defusers = [
                target for target in possible_targets
                if self.is_planted and target.defuse_timer > 0
            ]
            target_pool = defusers if defusers else possible_targets
            target = min(
                target_pool,
                key=lambda t: (
                    max(abs(t.pos[0] - shooter.pos[0]), abs(t.pos[1] - shooter.pos[1])),
                    t.hp,
                    t.name,
                ),
            )
            shot_intents.append({"shooter": shooter, "target": target})

        # シャッフル後に安定ソートすることで、同じ反応速度だけ順番がランダムになる。
        random.shuffle(shot_intents)
        shot_intents.sort(
            key=lambda intent: intent["shooter"].reaction,
            reverse=True,
        )

        for intent in shot_intents:
            shooter = intent["shooter"]
            target = intent["target"]

            if not shooter.is_alive:
                continue
            if not target.is_alive:
                continue

            shooter_accuracy = MOVING_ACCURACY if shooter.moved_this_tick else shooter.accuracy
            if shooter.blind_remaining > 0:
                shooter_accuracy *= BLIND_ACCURACY_MULTIPLIER

            effective_dodge = target.dodge_rate * (
                REVEALED_DODGE_MULTIPLIER
                if self._is_revealed_for_shot(target, current_los_revealed_names)
                else 1.0
            )
            hit_chance = shooter_accuracy * (1.0 - effective_dodge)
            if target.moved_this_tick:
                hit_chance *= MOVING_TARGET_HIT_MULTIPLIER
            hit_chance = max(0.0, min(1.0, hit_chance))

            hit = random.random() < hit_chance
            headshot = hit and random.random() < shooter.hs_rate
            damage = (HEADSHOT_DAMAGE if headshot else BODY_DAMAGE) if hit else 0

            shot = {
                "shooter": shooter,
                "target": target,
                "hit": hit,
                "headshot": headshot,
                "damage": damage,
                "hit_chance": hit_chance,
                "reaction": shooter.reaction,
            }
            executed_shots.append(shot)

            if damage > 0:
                target.hp = max(0, target.hp - damage)
                if target.hp <= 0:
                    self._kill_character(shooter, target)

        self.last_shots = executed_shots
        self.last_shot = executed_shots[-1] if executed_shots else None

    def _resolve_defuse_completion(self):
        """射撃後に解除完了を確定する。生存している解除者だけが完了できる。"""
        if not self.is_planted or self.is_defused:
            return
        completed = [
            c for c in self.chars
            if c.is_alive and c.team == "D" and c.defuse_timer >= DEFUSE_REQUIRED_SECONDS
        ]
        if completed:
            self.is_defused = True
            self.active_defuser_name = None

    def _explode_flash(self, projectile, impact=None):
        impact = impact or projectile["path"][min(projectile["progress"], len(projectile["path"]) - 1)]
        self.flash_bursts.append({"pos": impact, "remaining_seconds": FLASH_BURST_DURATION_SECONDS})
        owner_team = projectile.get("team")
        for char in self.chars:
            if not char.is_alive or char.team == owner_team:
                continue
            if self.check_cell_line_of_sight(tuple(char.pos), impact, block_smoke=True):
                char.blind_remaining = max(char.blind_remaining, BLIND_DURATION_SECONDS)

    def _explode_recon(self, projectile, impact=None):
        impact = impact or projectile["path"][min(projectile["progress"], len(projectile["path"]) - 1)]
        ir, ic = impact
        # 9x9: 着弾地点を中心に上下左右へ4マスずつ。
        radius = RECON_REVEAL_SIZE // 2
        cells = {
            (rr, cc)
            for rr in range(ir - radius, ir + radius + 1)
            for cc in range(ic - radius, ic + radius + 1)
            if 0 <= rr < self.height and 0 <= cc < self.width
        }
        self.recon_bursts.append({"cells": cells, "remaining_seconds": 1.0})
        owner_team = projectile.get("team")
        for char in self.chars:
            if char.is_alive and char.team != owner_team and tuple(char.pos) in cells:
                char.reveal_remaining = max(char.reveal_remaining, REVEAL_DURATION_SECONDS)

    def _advance_flash_projectiles(self):
        remaining = []
        for projectile in self.flash_projectiles:
            projectile["ticks_alive"] += 1
            next_progress = projectile["progress"] + FLASH_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(projectile["path"]) - 1
            projectile["progress"] = min(next_progress, len(projectile["path"]) - 1)
            if hit_wall_or_edge or projectile["ticks_alive"] >= FLASH_MAX_FLIGHT_TICKS:
                self._explode_flash(projectile)
            else:
                remaining.append(projectile)
        self.flash_projectiles = remaining

    def _advance_recon_projectiles(self):
        remaining = []
        for projectile in self.recon_projectiles:
            next_progress = projectile["progress"] + RECON_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(projectile["path"]) - 1
            projectile["progress"] = min(next_progress, len(projectile["path"]) - 1)
            if hit_wall_or_edge:
                self._explode_recon(projectile)
            else:
                remaining.append(projectile)
        self.recon_projectiles = remaining

    def process_battle(self):
        self.battle_tick += 1
        # 時間制効果は秒で管理する。TICK_TIMEを変更しても効果時間は変わらない。
        dt = self._tick_seconds()
        for char in self.chars:
            char.blind_remaining = max(0.0, char.blind_remaining - dt)
            char.reveal_remaining = max(0.0, char.reveal_remaining - dt)
        for burst in self.flash_bursts:
            burst["remaining_seconds"] -= dt
        self.flash_bursts = [burst for burst in self.flash_bursts if burst["remaining_seconds"] > 0]
        for burst in self.recon_bursts:
            burst["remaining_seconds"] -= dt
        self.recon_bursts = [burst for burst in self.recon_bursts if burst["remaining_seconds"] > 0]
        self._advance_flash_projectiles()
        self._advance_recon_projectiles()
        for smoke in self.smokes:
            smoke["remaining_seconds"] -= dt
        self.smokes = [smoke for smoke in self.smokes if smoke["remaining_seconds"] > 0]

        if self.spike_pos is not None:
            for c in self.chars:
                if c.is_alive and c.team == "A" and tuple(c.pos) == self.spike_pos:
                    c.has_spike = True
                    self.spike_pos = None
                    break

        for c in self.chars:
            if not c.is_alive and c.has_spike:
                self.spike_pos = tuple(c.pos)
                c.has_spike = False

        # 現在の射線状況は先に計算するが、射線リビール状態への反映は射撃後に行う。
        # そのため、初めて敵を視認したTickの射撃は通常の回避率で判定される。
        current_los_revealed_names = self._current_los_revealed_names()
        alive = [c for c in self.chars if c.is_alive]
        engagements = [
            (alive[i], alive[j])
            for i in range(len(alive))
            for j in range(i + 1, len(alive))
            if alive[i].team != alive[j].team and self.check_line_of_sight(alive[i], alive[j])
        ]
        self.last_engagements = engagements
        self.last_shot = None
        self._resolve_all_shots(engagements, current_los_revealed_names)

        # 射撃判定後に、現在の射線状況を次のTick用リビール状態として反映する。
        for char in self.chars:
            char.los_revealed = char.is_alive and char.name in current_los_revealed_names

        # 射撃結果で条件を満たした覚醒イベントを判定する。
        self._check_awakening_events()

        # 射撃を解決してから解除完了を判定する。
        self._resolve_defuse_completion()

        alive_A = any(c.is_alive for c in self.chars if c.team == "A")
        alive_D = any(c.is_alive for c in self.chars if c.team == "D")
        score_text = f" [Score: Att {self.attacker_wins} - {self.defender_wins} Def]"

        if self.is_defused:
            self.defender_wins += 1
            if not self.headless: self.label.config(text=f"⚙️ Spike Defused! Defender WIN Round {self.current_round}! {score_text}", fg="green")
            self.round_over = True
            self.check_match_winner()
        elif self.is_planted:
            self.detonate_timer -= self._tick_seconds()
            if self.detonate_timer <= 0:
                self.attacker_wins += 1
                if not self.headless: self.label.config(text=f"💥 Spike Detonated! Attacker WIN Round {self.current_round}! {score_text}", fg="red")
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                if not self.headless: self.label.config(text=f"🏆 Defender Annihilated! Attacker WIN Round {self.current_round}! {score_text}", fg="#c0392b")
                self.round_over = True
                self.check_match_winner()
            elif not alive_A:
                if not self.headless:
                    max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                    defuse_str = f" (Defusing: {max_defuse:.1f}/{DEFUSE_REQUIRED_SECONDS:.0f}s)" if max_defuse > 0 else ""
                    self.label.config(text=f"💀 Attacker Eliminated! Defuse the Spike! {math.ceil(self.detonate_timer)}s{defuse_str} | R{self.current_round}{score_text}", fg="#27ae60")
            elif not self.headless:
                max_defuse = max([c.defuse_timer for c in self.chars if c.team == "D" and c.is_alive] + [0])
                defuse_str = f" (Defusing: {max_defuse:.1f}/{DEFUSE_REQUIRED_SECONDS:.0f}s)" if max_defuse > 0 else ""
                self.label.config(text=f"🔥 Spike Planted! Detonation in {math.ceil(self.detonate_timer)}s{defuse_str} | R{self.current_round}{score_text}", fg="red")
        else:
            self.round_timer -= self._tick_seconds()
            if self.round_timer <= 0:
                self.defender_wins += 1
                if not self.headless: self.label.config(text=f"⏰ Time Expired! Defender WIN Round {self.current_round}! {score_text}", fg="#27ae60")
                self.round_over = True
                self.check_match_winner()
            elif not alive_A:
                self.defender_wins += 1
                if not self.headless: self.label.config(text=f"🏆 Attacker Annihilated! Defender WIN Round {self.current_round}! {score_text}", fg="#27ae60")
                self.round_over = True
                self.check_match_winner()
            elif not alive_D:
                self.attacker_wins += 1
                if not self.headless: self.label.config(text=f"🏆 Defender Annihilated! Attacker WIN Round {self.current_round}! {score_text}", fg="#c0392b")
                self.round_over = True
                self.check_match_winner()
            elif not self.headless:
                site_side = "Left Side" if self.target_plant_pos and self.target_plant_pos[1] < self.width // 2 else "Right Side"
                self.label.config(text=f"⚔️ Round {self.current_round} (Attacking {site_side}) | Ends in {math.ceil(self.round_timer)}s | {score_text}", fg="black")

    def _map_x(self, x):
        return self.map_offset_x + x

    def _draw_compact_ability_icon(self, ability_name, cx, cy, available):
        """構えるUIと同じ意匠の小型アビリティアイコンを描く。"""
        active_color = {"SMOKE": "#e67e22", "FLASH": "#f1c40f", "RECON": "#65d8e8", "HUNT": "#e74c3c"}[ability_name]
        fill = active_color if available else "#59616c"
        outline = "#f8c471" if available else "#7f8c8d"
        if ability_name == "SMOKE":
            for ox, oy, radius in [(-4, 2, 4), (0, -2, 5), (5, 2, 4), (1, 5, 4)]:
                self.canvas.create_oval(cx+ox-radius, cy+oy-radius, cx+ox+radius, cy+oy+radius,
                                        fill=fill, outline=outline, width=1)
        elif ability_name == "FLASH":
            self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill=fill, outline=outline, width=1)
            for dx, dy in [(0,-9),(0,9),(-9,0),(9,0)]:
                self.canvas.create_line(cx, cy, cx+dx, cy+dy, fill=fill, width=1)
        elif ability_name == "RECON":
            self.canvas.create_polygon(cx-8, cy+3, cx+5, cy-5, cx+8, cy-2, cx-4, cy+6,
                                       fill=fill, outline=outline)
        else:  # HUNT
            self.canvas.create_text(cx, cy, text="H", fill=fill, font=("Arial", 10, "bold"))

    def _draw_team_panel(self, team, x0, title, accent):
        """左右パネルへHP・K/D・アビリティ・現在の戦闘ステータスを表示する。"""
        panel_w = SIDE_PANEL_WIDTH
        panel_h = self.map_pixel_height + self.ability_area_height
        self.canvas.create_rectangle(
            x0, 0, x0 + panel_w, panel_h,
            fill="#111722", outline="#2a3444"
        )
        self.canvas.create_rectangle(x0, 0, x0 + panel_w, 42, fill=accent, outline="")
        self.canvas.create_text(
            x0 + panel_w / 2, 21,
            text=title, fill="white", font=("Arial", 13, "bold")
        )

        chars = [c for c in self.chars if c.team == team][:5]
        row_h = 108
        card_h = 100

        for i, char in enumerate(chars):
            y = 48 + i * row_h
            row_fill = "#1b2432" if char.is_alive else "#17191e"
            muted = "#aeb8c6" if char.is_alive else "#666b73"
            name_fill = "white" if char.is_alive else "#777b83"

            self.canvas.create_rectangle(
                x0 + 10, y, x0 + panel_w - 10, y + card_h,
                fill=row_fill, outline="#323e50"
            )

            # 名前・K/D
            self.canvas.create_text(
                x0 + 18, y + 13,
                text=char.display_name, anchor="w",
                fill=name_fill, font=("Arial", 9, "bold")
            )
            self.canvas.create_text(
                x0 + panel_w - 17, y + 13,
                text=f"K {char.kills}  D {char.deaths}", anchor="e",
                fill="#d6dde8" if char.is_alive else "#777b83",
                font=("Arial", 8, "bold")
            )

            # HPバー
            hp_ratio = char.hp / char.max_hp if char.is_alive and char.max_hp > 0 else 0.0
            bar_x1, bar_x2 = x0 + 18, x0 + panel_w - 18
            self.canvas.create_rectangle(bar_x1, y + 25, bar_x2, y + 37, fill="#343a45", outline="")
            self.canvas.create_rectangle(
                bar_x1, y + 25,
                bar_x1 + (bar_x2 - bar_x1) * hp_ratio, y + 37,
                fill=accent, outline=""
            )
            hp_text = f"HP {char.hp}/{char.max_hp}" if char.is_alive else "DEAD"
            self.canvas.create_text(
                (bar_x1 + bar_x2) / 2, y + 31,
                text=hp_text, fill="white", font=("Arial", 7, "bold")
            )

            # 現在値を表示するため、タイガー・コンボ・覚醒後の4能力から総合戦闘力も毎回再計算される。
            accuracy_pct = round(char.accuracy * 100)
            dodge_pct = round(char.dodge_rate * 100)
            hs_pct = round(char.hs_rate * 100)
            iq_text = f"{char.iq:g}"
            reaction_text = f"{char.reaction:g}"
            power_text = str(math.floor(char.combat_power))

            self.canvas.create_text(
                x0 + 18, y + 49,
                text=f"命中精度 {accuracy_pct}%   判断力(IQ) {iq_text}",
                anchor="w", fill=muted, font=("Arial", 7, "bold")
            )
            self.canvas.create_text(
                x0 + 18, y + 63,
                text=f"キャラコン(回避) {dodge_pct}%   HS {hs_pct}%",
                anchor="w", fill=muted, font=("Arial", 7, "bold")
            )
            self.canvas.create_text(
                x0 + 18, y + 77,
                text=f"反応速度 {reaction_text}",
                anchor="w", fill=muted, font=("Arial", 7, "bold")
            )
            self.canvas.create_text(
                x0 + panel_w - 18, y + 77,
                text=f"総合戦闘力 {power_text}",
                anchor="e", fill="#f5c76b" if char.is_alive else "#777b83",
                font=("Arial", 8, "bold")
            )

            # ロールに対応するアビリティ
            ability = char.ability_name
            charges = {
                "SMOKE": char.smoke_charges,
                "FLASH": char.flash_charges,
                "RECON": char.recon_charges,
                "HUNT": 1,
            }[ability]
            available = char.is_alive and (charges > 0 or ability == "HUNT")
            self._draw_compact_ability_icon(ability, x0 + 27, y + 91, available)
            label = {"SMOKE": "SMOKE", "FLASH": "FLASH", "RECON": "RECON", "HUNT": "HUNT +10%"}[ability]
            status = "PASSIVE" if ability == "HUNT" else f"残り {charges}"
            self.canvas.create_text(
                x0 + 43, y + 91,
                text=f"{label}  {status}", anchor="w",
                fill=muted, font=("Arial", 7, "bold")
            )

    def draw(self):
        if self.headless:
            return
        self.canvas.delete("all")
        self._draw_team_panel("A", 0, "ATTACKERS", "#c0392b")
        self._draw_team_panel("D", self.map_offset_x + self.map_pixel_width, "DEFENDERS", "#27ae60")

        color_map = {"0":"white", "1":"#34495e", "2":"#fff9c4", "3":"#ffcccc", "4":"#ccffcc"}
        for r in range(self.height):
            for c in range(self.width):
                x1 = self._map_x(c * self.cell_size)
                color = color_map.get(str(self.grid[r, c]), "white")
                self.canvas.create_rectangle(x1, r*self.cell_size, x1+self.cell_size, (r+1)*self.cell_size, fill=color, outline="#eee")

        # 煙らしく見えるように、半透明風の円を重ねて雲状に描画する。
        # Tkinter CanvasはRGBA非対応なのでstippleを使う。
        for smoke_index, smoke in enumerate(self.smokes):
            warning = smoke["remaining_seconds"] <= 3 * self._tick_seconds()
            if warning and self.battle_tick % 2 == 0:
                continue
            for sr, sc in smoke["cells"]:
                x1 = self._map_x(sc*self.cell_size)
                y1 = sr*self.cell_size
                pad = max(2, self.cell_size // 10)
                self.canvas.create_oval(
                    x1-pad, y1-pad, x1+self.cell_size+pad, y1+self.cell_size+pad,
                    fill="#d97706", outline="#f59e0b", width=1, stipple="gray50"
                )
                # 小さな煙の塊をずらして重ねる
                offsets = [
                    (-0.18, -0.12, 0.72), (0.20, -0.18, 0.66),
                    (-0.10, 0.22, 0.68), (0.24, 0.20, 0.62),
                ]
                for ox, oy, scale in offsets:
                    size = self.cell_size * scale
                    cx = x1 + self.cell_size * (0.5 + ox)
                    cy = y1 + self.cell_size * (0.5 + oy)
                    self.canvas.create_oval(
                        cx-size/2, cy-size/2, cx+size/2, cy+size/2,
                        fill="#f59e0b", outline="", stipple="gray50"
                    )

        if not self.is_planted and self.target_plant_pos:
            tr, tc = self.target_plant_pos
            x1 = self._map_x(tc*self.cell_size)
            self.canvas.create_rectangle(x1, tr*self.cell_size, x1+self.cell_size, (tr+1)*self.cell_size, fill="#f39c12", outline="#d35400")
        if self.is_planted and self.planted_pos:
            pr, pc = self.planted_pos
            x1 = self._map_x(pc*self.cell_size)
            self.canvas.create_oval(x1+2, pr*self.cell_size+2, x1+self.cell_size-2, (pr+1)*self.cell_size-2, fill="red", outline="")
        if self.spike_pos:
            sr, sc = self.spike_pos
            x1 = self._map_x(sc*self.cell_size)
            self.canvas.create_oval(x1+2, sr*self.cell_size+2, x1+self.cell_size-2, (sr+1)*self.cell_size-2, fill="black", outline="")

        if self.last_shot:
            shooter = self.last_shot["shooter"]
            target = self.last_shot["target"]
            self.canvas.create_line(
                self._map_x((shooter.pos[1]+0.5)*self.cell_size), (shooter.pos[0]+0.5)*self.cell_size,
                self._map_x((target.pos[1]+0.5)*self.cell_size), (target.pos[0]+0.5)*self.cell_size,
                fill="#ff3b30" if self.last_shot["hit"] else "#aab2bd", width=2, dash=() if self.last_shot["hit"] else (4, 3),
            )

        # 飛翔中フラッシュ：投擲済み経路を点線、現在位置を小さな光点で表示。
        for projectile in self.flash_projectiles:
            path = projectile["path"]
            end_index = min(projectile["progress"], len(path) - 1)
            if end_index > 0:
                coords = []
                for rr, cc in path[:end_index + 1]:
                    coords.extend([self._map_x((cc + 0.5) * self.cell_size), (rr + 0.5) * self.cell_size])
                if len(coords) >= 4:
                    self.canvas.create_line(*coords, fill="#f4d03f", width=2, dash=(3, 4))
            rr, cc = path[end_index]
            cx = self._map_x((cc + 0.5) * self.cell_size)
            cy = (rr + 0.5) * self.cell_size
            self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4, fill="#fff4a3", outline="#d4ac0d")

        # 飛翔中リコン。フラッシュ同様に点線で進行方向を示す。
        for projectile in self.recon_projectiles:
            path = projectile["path"]
            end_index = min(projectile["progress"], len(path) - 1)
            if end_index > 0:
                coords = []
                for rr, cc in path[:end_index + 1]:
                    coords.extend([self._map_x((cc + 0.5) * self.cell_size), (rr + 0.5) * self.cell_size])
                if len(coords) >= 4:
                    self.canvas.create_line(*coords, fill="#65d8e8", width=2, dash=(3, 4))
            rr, cc = path[end_index]
            cx = self._map_x((cc + 0.5) * self.cell_size)
            cy = (rr + 0.5) * self.cell_size
            self.canvas.create_polygon(cx-7, cy+3, cx+5, cy-5, cx+8, cy-2, cx-4, cy+6,
                                       fill="#9eeaf4", outline="#2aa9bd")

        for burst in self.recon_bursts:
            for rr, cc in burst["cells"]:
                x1 = self._map_x(cc * self.cell_size)
                y1 = rr * self.cell_size
                self.canvas.create_rectangle(x1, y1, x1+self.cell_size, y1+self.cell_size,
                                             fill="#6ed7e8", outline="", stipple="gray50")

        for burst in self.flash_bursts:
            rr, cc = burst["pos"]
            cx = self._map_x((cc + 0.5) * self.cell_size)
            cy = (rr + 0.5) * self.cell_size
            radius = self.cell_size * 0.65
            self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                                    fill="#fff7bf", outline="#f1c40f", width=2, stipple="gray50")

        selected_names = {ctrl.selected_char for ctrl, _ in self.get_user_controllers() if ctrl.selected_char is not None}
        viewer_team = self.get_viewer_team()
        visible_chars = [
            c for c in self.chars
            if self.is_visible_to_team(c, viewer_team) and (c.is_alive or c.just_died)
        ]
        for char in visible_chars:
            row, col = char.pos
            x1 = self._map_x(col*self.cell_size)
            cx, cy = x1 + self.cell_size/2, (row+0.5)*self.cell_size
            if char.just_died:
                self.canvas.create_text(cx, cy, text="X", fill="orange", font=("Arial", 12, "bold"))
                char.just_died = False
                continue

            bg = "#2980b9" if (char.defuse_timer > 0 and self.is_planted) else char.bg_color
            outline_color = "yellow" if char.name in selected_names else ""
            outline_width = 3 if char.name in selected_names else 1
            self.canvas.create_oval(x1, row*self.cell_size, x1+self.cell_size, (row+1)*self.cell_size, fill=bg, outline=outline_color, width=outline_width)
            if char.blind_remaining > 0:
                # 視認性を壊さない薄い二重リングと小さな印でブラインド状態を表示。
                self.canvas.create_oval(x1+2, row*self.cell_size+2, x1+self.cell_size-2, (row+1)*self.cell_size-2,
                                        outline="#fff2a8", width=2, dash=(2, 2))
                self.canvas.create_text(cx, cy, text="✦", fill="#fff7c2", font=("Arial", 9, "bold"))
            if self._is_revealed(char):
                self.canvas.create_rectangle(x1+3, row*self.cell_size+3, x1+self.cell_size-3, (row+1)*self.cell_size-3,
                                             outline="#7de3f2", width=2, dash=(4, 2))
                self.canvas.create_text(cx, cy+7, text="◇", fill="#b8f4fb", font=("Arial", 8, "bold"))
            if char.has_spike:
                self.canvas.create_oval(x1+3, row*self.cell_size+3, x1+self.cell_size-3, (row+1)*self.cell_size-3, fill="black", outline="")

            # キャラクター上部の名前・HPパネル
            panel_w = max(56, min(150, 16 + len(char.display_name) * 7))
            panel_h = 23
            px1 = cx - panel_w / 2
            py2 = row*self.cell_size - 3
            py1 = py2 - panel_h
            has_adjacent = any(
                other is not char and other.is_alive
                and max(abs(other.pos[0] - char.pos[0]), abs(other.pos[1] - char.pos[1])) <= 1
                for other in visible_chars
            )
            # 隣接時は札を半透明風(stipple)にして重なりの圧迫感を減らす。
            self.canvas.create_rectangle(px1, py1, px1+panel_w, py2, fill="#101820", outline=char.bg_color,
                                         width=1, stipple="gray50" if has_adjacent else "")
            name_color = "#d6d9de" if has_adjacent else ("yellow" if char.has_spike else "white")
            self.canvas.create_text(cx, py1+8, text=char.display_name, fill=name_color, font=("Arial", 8, "bold"))
            hp_ratio = char.hp / char.max_hp
            self.canvas.create_rectangle(px1+4, py2-7, px1+panel_w-4, py2-3, fill="#3a404a", outline="")
            self.canvas.create_rectangle(px1+4, py2-7, px1+4+(panel_w-8)*hp_ratio, py2-3, fill=char.bg_color, outline="")

        # アビリティ専用の下部領域。マップとは完全に分離する。
        bottom_y = self.map_pixel_height
        self.canvas.create_rectangle(
            self.map_offset_x, bottom_y,
            self.map_offset_x + self.map_pixel_width, bottom_y + self.ability_area_height,
            fill="#0b1018", outline="#2a3444", width=2
        )
        self.canvas.create_text(
            self.map_offset_x + 18, bottom_y + 18,
            text="ABILITIES", anchor="w", fill="#8b98a9", font=("Arial", 10, "bold")
        )

        bounds = self._ability_panel_bounds()
        selected = self._selected_user_character()
        if bounds and selected:
            ability_name = selected.ability_name
            x1, y1, x2, y2 = bounds
            accent = {"SMOKE": "#e67e22", "FLASH": "#f1c40f", "RECON": "#65d8e8", "HUNT": "#e74c3c"}[ability_name]
            charges = {
                "SMOKE": selected.smoke_charges,
                "FLASH": selected.flash_charges,
                "RECON": selected.recon_charges,
                "HUNT": 1,
            }[ability_name]
            armed = self.ability_mode == (ability_name, selected.team, selected.name)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#151c27",
                                         outline=accent if armed or ability_name == "HUNT" else "#536273", width=2)
            icon_cx, icon_cy = x1 + 42, (y1 + y2) / 2
            self._draw_compact_ability_icon(ability_name, icon_cx, icon_cy, charges > 0)

            text_cx = (x1 + 70 + x2) / 2
            if ability_name == "HUNT":
                state = "HUNT / ハンター（常時発動）"
                help_text = "Hit% +10ポイント・HS% +5ポイント"
            else:
                label = {"SMOKE": "SMOKE", "FLASH": "FLASH", "RECON": "RECON"}[ability_name]
                state = "構え中：再クリックでキャンセル" if armed else f"{label}  残り {charges}"
                help_text = ("方向を指定" if armed and ability_name in ("FLASH", "RECON")
                             else ("マスを選択" if armed else f"クリックして{label}を構える"))
            self.canvas.create_text(text_cx, y1+22, text=state,
                                    fill=accent if charges else "#777", font=("Arial", 10, "bold"))
            self.canvas.create_text(text_cx, y1+48, text=help_text, fill="white", font=("Arial", 9))

            plant_bounds = self._plant_button_bounds()
            if plant_bounds:
                px1, py1, px2, py2 = plant_bounds
                planting = selected.is_planting
                self.canvas.create_rectangle(px1, py1, px2, py2, fill="#2a1d0d",
                                             outline="#f39c12" if planting else "#8a6a32", width=2)
                self.canvas.create_text((px1+px2)/2, py1+22,
                                        text="PLANTING..." if planting else "PLANT",
                                        fill="#ffd27a", font=("Arial", 11, "bold"))
                progress = min(1.0, selected.plant_timer / max(0.001, PLANT_REQUIRED_SECONDS))
                self.canvas.create_rectangle(px1+14, py2-20, px2-14, py2-12, fill="#4b3a22", outline="")
                self.canvas.create_rectangle(px1+14, py2-20, px1+14+(px2-px1-28)*progress, py2-12,
                                             fill="#f39c12", outline="")

        # 既存ゲーム画面を下へずらし、告知がマップへ重ならない専用領域を確保する。
        self.canvas.move("all", 0, COMBO_BANNER_HEIGHT)
        self._draw_combo_announcement_banner()

    def _draw_combo_announcement_banner(self):
        """コンボと覚醒イベントを共通の上部パネルへ描画する。"""
        total_w = self.map_pixel_width + SIDE_PANEL_WIDTH * 2
        self.canvas.create_rectangle(0, 0, total_w, COMBO_BANNER_HEIGHT, fill="#090d14", outline="#2a3444", width=2)

        queue = getattr(self, "announcement_queue", [])
        if self.combo_announcement_index >= len(queue):
            self.canvas.create_text(
                total_w / 2, COMBO_BANNER_HEIGHT / 2,
                text=f"ROUND {self.current_round}", fill="#768394",
                font=("Arial", 14, "bold")
            )
            return

        announcement = queue[self.combo_announcement_index]
        is_awakening = announcement.get("type") == "awakening"
        team = announcement.get("team")
        accent = "#c0392b" if team == "A" else "#27ae60"
        team_text = "ATTACKERS" if team == "A" else "DEFENDERS"
        category_text = "覚醒イベント" if is_awakening else "プレイヤーコンボ"
        title_color = "#ff8f70" if is_awakening else "#ffd66b"

        self.canvas.create_rectangle(12, 10, total_w-12, COMBO_BANNER_HEIGHT-10, fill="#151c27", outline=accent, width=3)
        self.canvas.create_text(
            30, 25, text=category_text, anchor="w",
            fill=accent, font=("Arial", 10, "bold")
        )
        self.canvas.create_text(
            total_w/2, 25, text=announcement.get("name", "名称未設定"),
            fill=title_color, font=("Arial", 17, "bold")
        )
        names = " × ".join(announcement.get("display_players", announcement.get("players", ())))
        self.canvas.create_text(
            total_w/2, 51, text=f"{team_text}  |  {names}",
            fill="white", font=("Arial", 11, "bold")
        )
        self.canvas.create_text(
            total_w/2, 73, text=announcement.get("effect_text", "特殊効果"),
            fill="#b9c6d8", font=("Arial", 10)
        )

    def loop(self):
        if not self.round_over and not self.match_over:
            for c in self.chars:
                if c.is_alive: self.move_character(c)
            self.process_battle()
            self.draw()
            self._advance_combo_announcement()
            self.root.after(TICK_TIME, self.loop)

    def run_headless_loop(self):
        """【AI学習用】画面を描画せず、限界速度でシミュレーションを回す"""
        print("💡 Headless Mode: シミュレーションをバックグラウンドで高速実行中...")
        while not self.match_over:
            if not self.round_over:
                for c in self.chars:
                    if c.is_alive: self.move_character(c)
                self.process_battle()
                self._advance_combo_announcement()
            # round_over 時の初期化は check_match_winner 内で自動処理されます

    def run(self):
        if self.headless:
            self.run_headless_loop()
        else:
            self.draw()
            self.root.after(TICK_TIME, self.loop)
            self.root.mainloop()

if __name__ == "__main__":
    # =========================================================================
    # ⚙️ モード切り替えスイッチ
    # =========================================================================
    # TRUE : 画面なしでバックグラウンド超高速計算（学習やテスト用）
    # FALSE: 画面ありでいつものプレイ（人間が観戦する用）
    LEARNING_MODE = True # AIモデルを使うのでTrueに

    def start_match(roster, spike_holder_name=None):
        """💡編成画面で「決定」が押されたときに呼ばれる。rosterは選ばれたキャラ名5人分のリスト"""
        #att_ctrl = DefaultAttackerController()
        att_ctrl = UserInputController()

        if LEARNING_MODE:
            # 新しい統合モデルでテストしたい場合
            def_ctrl = LearningDefenderAllAIController(model_path="dqn_defender_combined_best.pt")

            # 迷路探索のみモデル
            #def_ctrl = LearningDefenderController(model_path="dqn_gridworld_fixedmap.pt")

            # 動きを確認したいので headless=False で可視化する
            game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False, attacker_roster=roster, spike_holder_name=spike_holder_name)
        else:
            def_ctrl = DefaultDefenderController()
            game = VisualFPSBattle(NEW_MAZE_STR, att_ctrl, def_ctrl, headless=False, attacker_roster=roster, spike_holder_name=spike_holder_name)

        game.run()

    # 💡まずキャラクター編成画面を開き、「決定」が押されたら start_match が呼ばれて試合開始
    roster_screen = RosterSelectScreen(on_confirm=start_match)
    roster_screen.run()