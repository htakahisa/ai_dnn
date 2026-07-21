# data_loader.py
"""外部データ（キャラクター・コンボ・覚醒イベント）の読み込みと、
関連する数値計算ロジックをまとめたモジュール。"""

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