from flask import redirect, render_template, request, url_for
import json
from pathlib import Path
from datetime import datetime, timezone
import sys

if __package__:
    from .analysis_branching import (
        AI_ANALYSIS_RULES,
        AnalysisUnavailableError,
        analyze_match_data,
    )
    from .match_calculation import calculate_match_data_from_original
else:
    from analysis_branching import (
        AI_ANALYSIS_RULES,
        AnalysisUnavailableError,
        analyze_match_data,
    )
    from match_calculation import calculate_match_data_from_original

ANALYTICS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYTICS_DIR.parent
SERIES_DATA_DIR = PROJECT_ROOT / "series_data"
TRAINING_LABELS_PATH = ANALYTICS_DIR / "training_labels.jsonl"
COMPETITION_RESULTS_DIR = PROJECT_ROOT / "competition_results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from map_data import NEW_MAZE_STR

    MAP_GRID = [
        [int(cell) for cell in row.strip()]
        for row in NEW_MAZE_STR.strip().splitlines()
        if row.strip()
    ]
except (ImportError, ValueError):
    MAP_GRID = []


def _load_replay_frames(original_data):
    """Load state frames from the source result when the web copy omitted them."""
    maps = original_data.get("maps") or []
    if maps and maps[0].get("replay_frames"):
        return maps[0]["replay_frames"]

    team1 = original_data.get("team1")
    team2 = original_data.get("team2")
    map_data = maps[0] if maps else {}
    score1 = map_data.get("score1")
    score2 = map_data.get("score2")
    candidates = []
    file_pattern = (
        f"series_{str(team1).replace(' ', '_')}_vs_"
        f"{str(team2).replace(' ', '_')}_{original_data.get('team1_score', 0)}-"
        f"{original_data.get('team2_score', 0)}_*.json"
    )
    for path in COMPETITION_RESULTS_DIR.glob(file_pattern):
        try:
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
            if result.get("team1") != team1 or result.get("team2") != team2:
                continue
            for result_map in result.get("maps", []):
                if (
                    result_map.get("score1") == score1
                    and result_map.get("score2") == score2
                    and result_map.get("replay_frames")
                ):
                    candidates.append((path.stat().st_mtime, result_map["replay_frames"]))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return max(candidates, default=(0, []), key=lambda item: item[0])[1]


def _load_match_for_path(match_path):
    """ラベル保存時にも、表示時と同じ計算済み特徴量を作る。"""
    full_match_path = (SERIES_DATA_DIR / match_path).resolve()
    if not full_match_path.is_relative_to(SERIES_DATA_DIR.resolve()):
        raise ValueError("Invalid match path")
    original_path = full_match_path.with_name(
        full_match_path.name.replace("_inference.json", "_original.json")
    )
    if original_path.exists():
        with original_path.open("r", encoding="utf-8") as file:
            return calculate_match_data_from_original(json.load(file))
    with full_match_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_training_labels(match_path):
    """画面で入力したチーム別ラベルを、XGBoost学習用JSONLとして保存する。"""
    try:
        match_data = _load_match_for_path(match_path)
        team_names = (
            match_data["match_metadata"]["team1"],
            match_data["match_metadata"]["team2"],
        )
        for team_name in team_names:
            labels = {}
            for rule in AI_ANALYSIS_RULES:
                field = f"{team_name}__{rule['variable']}"
                if rule["type"] == "bool":
                    labels[rule["variable"]] = field in request.form
                else:
                    value = request.form.get(
                        field,
                        rule.get("default", rule["enum_values"][0]),
                    )
                    if value not in rule["enum_values"]:
                        return f"Invalid enum value: {rule['variable']}", 400
                    labels[rule["variable"]] = value

            team_data = dict(match_data)
            team_data["map_aggregate"] = match_data["map_aggregate"][team_name]
            team_data["player_stats"] = [
                player
                for player in match_data["player_stats"]
                if player.get("team") == team_name
            ]
            record = {
                "match_path": match_path,
                "team_name": team_name,
                "features": team_data,
                "labels": labels,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            }
            TRAINING_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TRAINING_LABELS_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return redirect(url_for("match_detail", match_path=match_path))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return f"ラベル保存に失敗しました: {exc}", 400


def match_detail(match_path):
    """個別試合の詳細ページ：originalから計算したデータをinference.jsonに書き出して0値を解消"""
    try:
        full_match_path = (SERIES_DATA_DIR / match_path).resolve()
        if not full_match_path.is_relative_to(SERIES_DATA_DIR.resolve()):
            return "Invalid match path", 400
        # まずoriginal.jsonのパスを取得（同じフォルダ内の_original.jsonを読み込む）
        original_path = full_match_path.with_name(
            full_match_path.name.replace("_inference.json", "_original.json")
        )
        if original_path.exists():
            with open(original_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)
            # Replay data has existed in two shapes.  Older files expose
            # ``rounds`` directly, while competition output stores the same
            # information in ``maps[0].round_records``.  Keep the template's
            # replay contract stable for both formats.
            if "rounds" not in original_data:
                first_map = (original_data.get("maps") or [{}])[0]
                round_records = []
                initial_attacker = first_map.get("initial_attacker")
                for index, record in enumerate(first_map.get("round_records", [])):
                    round_record = dict(record)
                    if initial_attacker:
                        attacker_team = (
                            initial_attacker
                            if index % 2 == 0
                            else (
                                original_data["team2"]
                                if initial_attacker == original_data["team1"]
                                else original_data["team1"]
                            )
                        )
                        round_record.setdefault("attacker_team", attacker_team)
                        round_record.setdefault(
                            "defender_team",
                            original_data["team2"]
                            if attacker_team == original_data["team1"]
                            else original_data["team1"],
                        )
                    round_records.append(round_record)
                original_data = {
                    **original_data,
                    "rounds": round_records,
                    "players": first_map.get("player_stats", []),
                }
            # originalから計算した正しいデータを生成
            match_data = calculate_match_data_from_original(original_data)
            # 【重要】計算したデータをinference.jsonに書き出して保存（0値問題を解消）
            with open(full_match_path, "w", encoding="utf-8") as f:
                json.dump(match_data, f, ensure_ascii=False, indent=2)
            print(f"inference.json updated: {full_match_path}")
        else:
            # 万が一originalがない場合は既存のinferenceを読み込む
            with open(full_match_path, "r", encoding="utf-8") as f:
                match_data = json.load(f)

        replay_frames = (
            _load_replay_frames(original_data) if "original_data" in locals() else []
        )

        # チーム別にAI分析結果を生成
        team1 = match_data["match_metadata"]["team1"]
        team2 = match_data["match_metadata"]["team2"]
        # 各チームのデータを正しくマージして分析を実行（map_aggregate内のチーム別データを優先）
        team1_full_data = match_data.copy()
        team1_full_data["map_aggregate"] = match_data["map_aggregate"][team1]
        team1_full_data["player_stats"] = [
            p for p in match_data["player_stats"] if p["team"] == team1
        ]

        team2_full_data = match_data.copy()
        team2_full_data["map_aggregate"] = match_data["map_aggregate"][team2]
        team2_full_data["player_stats"] = [
            p for p in match_data["player_stats"] if p["team"] == team2
        ]

        # 不足しているラベルを確認
        required_variables = {rule["variable"] for rule in AI_ANALYSIS_RULES}
        team1_missing = []
        team2_missing = []
        label_keys = {team1: set(), team2: set()}
        if TRAINING_LABELS_PATH.exists():
            with TRAINING_LABELS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if (
                            record["match_path"] == match_path
                            and record["team_name"] == team1
                        ):
                            label_keys[team1].update(record.get("labels", {}).keys())
                        elif (
                            record["match_path"] == match_path
                            and record["team_name"] == team2
                        ):
                            label_keys[team2].update(record.get("labels", {}).keys())
                    except Exception as e:
                        print(f"Error parsing training label: {e}")
        team1_missing = list(required_variables - label_keys[team1])
        team2_missing = list(required_variables - label_keys[team2])

        match_data["ai_analysis"] = {}
        match_data["ai_analysis_errors"] = {}
        for team_name, team_data in (
            (team1, team1_full_data),
            (team2, team2_full_data),
        ):
            try:
                match_data["ai_analysis"][team_name] = analyze_match_data(
                    team_data, team_name
                )
            except AnalysisUnavailableError as exc:
                match_data["ai_analysis"][team_name] = []
                match_data["ai_analysis_errors"][team_name] = str(exc)
        # original_dataもテンプレートに渡してリプレイ再生を可能にする
        return render_template(
            "match.html",
            match=match_data,
            match_path=match_path,
            analysis_rules=AI_ANALYSIS_RULES,
            missing_labels={team1: team1_missing, team2: team2_missing},
            original_data=original_data if "original_data" in locals() else None,
            replay_frames=replay_frames,
            map_grid=MAP_GRID,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Error loading match: {str(e)}", 500


def get_missing_label_stats():
    """training_labels.jsonlから不足しているラベル項目を持つ試合を抽出"""
    required_variables = {rule["variable"] for rule in AI_ANALYSIS_RULES}
    labels_by_match = {}

    if TRAINING_LABELS_PATH.exists():
        with TRAINING_LABELS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = (record["match_path"], record["team_name"])
                    labels_by_match.setdefault(key, set()).update(
                        record.get("labels", {}).keys()
                    )
                except Exception as e:
                    print(f"Error parsing training label: {e}")
    return [
        {
            "match_path": match_path,
            "team_name": team_name,
            "missing_labels": list(required_variables - labels),
            "labeled_at": "",
        }
        for (match_path, team_name), labels in labels_by_match.items()
        if required_variables - labels
    ]


def get_all_matches():
    """series_data内の全試合一覧を取得"""
    matches = []
    if SERIES_DATA_DIR.exists():
        for folder in SERIES_DATA_DIR.iterdir():
            if folder.is_dir():
                # original.jsonを優先的に探す
                original_file = next(
                    (f for f in folder.iterdir() if f.name.endswith("_original.json")),
                    None,
                )
                if original_file:
                    try:
                        with open(original_file, "r", encoding="utf-8") as f:
                            original_data = json.load(f)
                        # メタデータだけ抽出して一覧用に整形
                        matches.append(
                            {
                                "folder_name": folder.name,
                                "metadata": {
                                    "team1": original_data["team1"],
                                    "team2": original_data["team2"],
                                    "team1_score": original_data["team1_score"],
                                    "team2_score": original_data["team2_score"],
                                    "total_rounds": sum(
                                        m["score1"] + m["score2"]
                                        for m in original_data.get("maps", [])
                                    ),
                                    "timestamp": datetime.fromtimestamp(
                                        folder.stat().st_mtime, timezone.utc
                                    ).isoformat(),
                                },
                                "path": (
                                    original_file.relative_to(SERIES_DATA_DIR)
                                    .as_posix()
                                    .replace("_original.json", "_inference.json")
                                ),
                            }
                        )
                    except Exception as e:
                        print(f"Error loading {folder.name}: {e}")
    return sorted(
        matches, key=lambda x: x["metadata"].get("timestamp", ""), reverse=True
    )


# お気に入りを保存するファイルパス
FAVORITES_PATH = Path(__file__).parent / "favorites.json"


def save_favorite(match_path):
    """試合をお気に入りに登録/解除する"""
    try:
        favorites = []
        if FAVORITES_PATH.exists():
            with FAVORITES_PATH.open("r", encoding="utf-8") as f:
                favorites = json.load(f)

        # 既存のお気に入りにあるか確認
        if match_path in favorites:
            favorites.remove(match_path)
        else:
            favorites.append(match_path)

        # 保存
        with FAVORITES_PATH.open("w", encoding="utf-8") as f:
            json.dump(favorites, f, ensure_ascii=False)

        return redirect(url_for("index"))
    except Exception as e:
        return f"お気に入りの保存に失敗しました: {e}", 400


def get_favorites():
    """お気に入りの試合パス一覧を取得"""
    if FAVORITES_PATH.exists():
        with FAVORITES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def index():
    """トップページ：試合一覧表示＋不足項目のある試合を表示"""
    matches = get_all_matches()
    missing_labels = get_missing_label_stats()
    favorites = get_favorites()
    return render_template(
        "index.html",
        matches=matches,
        missing_labels=missing_labels,
        favorites=favorites,
    )
