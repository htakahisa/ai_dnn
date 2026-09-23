def calculate_match_data_from_original(original_data):
    """original.jsonの生データからmap_aggregateやround_featuresを計算して補完（ダミーテンプレート不使用、チーム別に集計）"""
    team1 = original_data["team1"]
    team2 = original_data["team2"]

    # チーム別統計を全てteam_tacticsに統合（プレイヤーのside情報を活用）
    team_tactics = {
        team1: {
            "entry_fights": 0,
            "entry_wins": 0,
            "attacker_rounds": 0,
            "attacker_wins": 0,
            "defender_rounds": 0,
            "defender_wins": 0,
            "post_plant_wins": 0,
            "plant_count": 0,
            "retake_attempts": 0,
            "retake_wins": 0,
            "time_expired_count": 0,
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "plants": 0,
            "defuses": 0,
            "first_kills": 0,
            "first_deaths": 0,
            "one_v_one_participated": 0,
            "one_v_one_won": 0,
            "ability_assists": 0,
            "ability_uses": 0,
            "attack_strategies": set(),
            "attack_sites": set(),
            "preaim_angle_sum": 0.0,
            "preaim_angle_count": 0,
        },
        team2: {
            "entry_fights": 0,
            "entry_wins": 0,
            "attacker_rounds": 0,
            "attacker_wins": 0,
            "defender_rounds": 0,
            "defender_wins": 0,
            "post_plant_wins": 0,
            "plant_count": 0,
            "retake_attempts": 0,
            "retake_wins": 0,
            "time_expired_count": 0,
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "plants": 0,
            "defuses": 0,
            "first_kills": 0,
            "first_deaths": 0,
            "one_v_one_participated": 0,
            "one_v_one_won": 0,
            "ability_assists": 0,
            "ability_uses": 0,
            "attack_strategies": set(),
            "attack_sites": set(),
            "preaim_angle_sum": 0.0,
            "preaim_angle_count": 0,
        },
    }

    # 全ラウンドの情報を抽出
    all_rounds = []

    # 各マップのプレイヤースタッツとラウンド情報を直接original.jsonから抽出
    for map_data in original_data.get("maps", []):
        # チーム別スタッツを合算（original.jsonのplayer_statsから直接取得）
        # original.jsonから直接round_recordsを取得（外部テンプレート不使用）
        original_rounds = map_data.get("round_records", [])
        if not original_rounds:
            continue

        # originalのラウンドデータから直接特徴量を抽出（original.jsonのround_recordsの構造に完全に合わせる）
        for r_idx, r in enumerate(original_rounds):
            # 設置の有無を分母に使い、勝敗はラウンドの勝者から判定する。
            reason = r.get("reason", "")
            tactic = r.get("tactic", {})
            attacker_strategy = tactic.get("attacker_strategy")
            planted = r.get("planted", False)

            winner = r.get("winner")
            is_attacker_win = (
                winner == "attacker"
                if winner in ("attacker", "defender")
                else reason in ("detonated", "defender_wipe")
            )
            entry_attempted = attacker_strategy in ["split", "default"]
            post_plant_occurred = bool(planted)
            retake_attempted = bool(planted)
            is_time_expired = reason == "time_expired"

            # ラウンド内のプレイヤー情報から攻撃側/防衛側のチームを特定
            players = r.get("players", {})
            attacker_teams = set()
            defender_teams = set()
            for p_name, p_data in players.items():
                p_team = p_data.get("team", "")
                p_side = p_data.get("side", "")
                if p_side == "attacker":
                    attacker_teams.add(p_team)
                elif p_side == "defender":
                    defender_teams.add(p_team)
                # キル、デス、アシストをチーム別に加算
                if p_team in team_tactics:
                    team_tactics[p_team]["kills"] += p_data.get("kills", 0)
                    team_tactics[p_team]["deaths"] += p_data.get("deaths", 0)
                    team_tactics[p_team]["assists"] += p_data.get("assists", 0)
                    team_tactics[p_team]["first_kills"] += p_data.get("first_kills", 0)
                    team_tactics[p_team]["first_deaths"] += p_data.get("first_deaths", 0)
                    team_tactics[p_team]["one_v_one_participated"] += p_data.get("one_v_one_participated", 0)
                    team_tactics[p_team]["one_v_one_won"] += p_data.get("one_v_one_won", 0)
                    team_tactics[p_team]["preaim_angle_sum"] += float(p_data.get("preaim_angle_sum", 0.0))
                    team_tactics[p_team]["preaim_angle_count"] += int(p_data.get("preaim_angle_count", 0))
            # 攻撃側・防衛側のチーム名を取得（基本的に各ラウンドで1チームずつ）
            attacker_team_name = next(iter(attacker_teams), None)
            if attacker_team_name not in team_tactics and defender_teams:
                observed_defender = next(iter(defender_teams))
                if observed_defender in team_tactics:
                    attacker_team_name = team2 if observed_defender == team1 else team1
            if attacker_team_name not in team_tactics:
                attacker_team_name = tactic.get("attacker_team")
            if attacker_team_name not in team_tactics:
                initial_attacker = map_data.get("initial_attacker", team1)
                if initial_attacker not in team_tactics:
                    initial_attacker = team1
                other_team = team2 if initial_attacker == team1 else team1
                round_number = int(r.get("round_number", r_idx + 1))
                attacker_team_name = (
                    initial_attacker
                    if round_number <= 12
                    or (round_number >= 25 and round_number % 2 == 1)
                    else other_team
                )
            defender_team_name = team2 if attacker_team_name == team1 else team1

            # 実際にそのサイドで戦ったラウンドの勝率を記録する。
            if attacker_team_name in team_tactics:
                team_tactics[attacker_team_name]["attacker_rounds"] += 1
                if is_attacker_win:
                    team_tactics[attacker_team_name]["attacker_wins"] += 1
            if defender_team_name in team_tactics:
                team_tactics[defender_team_name]["defender_rounds"] += 1
                if not is_attacker_win:
                    team_tactics[defender_team_name]["defender_wins"] += 1

            # 攻撃側のエントリー戦術メトリクスを加算
            if entry_attempted and attacker_team_name in team_tactics:
                team_tactics[attacker_team_name]["entry_fights"] += 1
                if is_attacker_win:
                    team_tactics[attacker_team_name]["entry_wins"] += 1
            # 設置した全ラウンドを数え、そのうち攻撃側が勝った回数を記録する。
            if post_plant_occurred and attacker_team_name in team_tactics:
                team_tactics[attacker_team_name]["plant_count"] += 1
                team_tactics[attacker_team_name]["plants"] += 1
                if is_attacker_win:
                    team_tactics[attacker_team_name]["post_plant_wins"] += 1
            # 設置された全ラウンドを防衛側のリテイク対象として数える。
            if retake_attempted and defender_team_name in team_tactics:
                team_tactics[defender_team_name]["retake_attempts"] += 1
                if not is_attacker_win:
                    team_tactics[defender_team_name]["retake_wins"] += 1
                if reason == "defused":
                    team_tactics[defender_team_name]["defuses"] += 1
            # 時間切れを攻撃側に加算
            if is_time_expired and attacker_team_name in team_tactics:
                team_tactics[attacker_team_name]["time_expired_count"] += 1

            # サイト情報をtactic.final_attack_siteから取得
            site = tactic.get("final_attack_site")

            if attacker_team_name in team_tactics:
                if attacker_strategy:
                    team_tactics[attacker_team_name]["attack_strategies"].add(attacker_strategy)
                if site:
                    team_tactics[attacker_team_name]["attack_sites"].add(site)

            # 勝者チームを特定
            winner_team = attacker_team_name if is_attacker_win else defender_team_name

            all_rounds.append(
                {
                    "round_number": len(all_rounds) + 1,
                    "attacker_team": attacker_team_name,
                    "defender_team": defender_team_name,
                    "winner": "attacker" if is_attacker_win else "defender",
                    "winner_team": winner_team,
                    "entry_attempted": entry_attempted,
                    "post_plant_occurred": post_plant_occurred,
                    "retake_attempted": retake_attempted,
                    "site": site,
                    "attack": attacker_strategy,
                    "result": reason,
                    "team_a_kills": r.get("team_a_kills", 0),
                    "team_d_kills": r.get("team_d_kills", 0),
                    "length": r.get("length", 60),
                }
            )

    # 各チームの最低限のデータを保証（ゼロ除算防止）
    for team in [team1, team2]:
        if team_tactics[team]["entry_fights"] == 0:
            team_tactics[team]["entry_winrate"] = 0.0
        if team_tactics[team]["plant_count"] == 0:
            team_tactics[team]["post_plant_winrate"] = 0.0
        if team_tactics[team]["retake_attempts"] == 0:
            team_tactics[team]["retake_winrate"] = 0.0
        if team_tactics[team]["attacker_rounds"] == 0:
            team_tactics[team]["attacker_winrate"] = 0.0
        if team_tactics[team]["defender_rounds"] == 0:
            team_tactics[team]["defender_winrate"] = 0.0

    # プレイヤー単位で取得できるミクロ指標をチームへ集約する。
    for map_data in original_data.get("maps", []):
        players_by_name = {
            p.get("name"): p.get("team") for p in map_data.get("player_stats", [])
        }
        for event in map_data.get("assist_events", []):
            team = players_by_name.get(event.get("assister"))
            if team in team_tactics and event.get("method") in {"flash", "recon", "smoke"}:
                team_tactics[team]["ability_assists"] += 1
        previous = {}
        for frame in map_data.get("replay_frames", []):
            for char in frame.get("chars", []):
                name = char.get("name")
                team = char.get("team")
                charges = int(char.get("ability_charges", 0))
                if name in previous and team in team_tactics:
                    team_tactics[team]["ability_uses"] += max(0, previous[name] - charges)
                previous[name] = charges

    for team in [team1, team2]:
        t = team_tactics[team]
        rounds = max(1, t["attacker_rounds"] + t["defender_rounds"])
        t["attack_strategy_variety"] = len(t.pop("attack_strategies"))
        t["attack_site_variety"] = len(t.pop("attack_sites"))
        t["kill_death_ratio"] = t["kills"] / max(1, t["deaths"])
        t["first_death_rate"] = t["first_deaths"] / rounds
        t["one_v_one_winrate"] = t["one_v_one_won"] / max(1, t["one_v_one_participated"])
        t["ability_assist_rate"] = t["ability_assists"] / max(1, t["kills"])
        t["ability_use_rate"] = t["ability_uses"] / rounds
        t["preaim_angle_mean"] = t["preaim_angle_sum"] / max(1, t["preaim_angle_count"])
        # This value is an angular error, not a positive "aim quality" score.
        # Smaller values therefore indicate better pre-aim alignment.
        t["preaim_error_mean"] = t["preaim_angle_mean"]

    ability_data_available = int(
        any(
            "ability_charges" in char
            for map_data in original_data.get("maps", [])
            for frame in map_data.get("replay_frames", [])
            for char in frame.get("chars", [])
        )
    )
    for team in [team1, team2]:
        team_tactics[team]["ability_data_available"] = ability_data_available

    # プレイヤー別のスタッツを整形
    player_stats = []
    for map_data in original_data.get("maps", []):
        for p in map_data.get("player_stats", []):
            player_stats.append(
                {
                    "name": p["name"],
                    "team": p["team"],
                    "kd": p["kills"] / p["deaths"] if p["deaths"] > 0 else 0.0,
                    "firstk": p.get("first_kills", 0),
                    "firstd": p.get("first_deaths", 0),
                    "one_v_one_participated": p.get("one_v_one_participated", 0),
                    "1v1_winrate": (
                        p["one_v_one_won"] / p["one_v_one_participated"]
                        if p["one_v_one_participated"] > 0
                        else 0.0
                    ),
                    "preaim_angle_sum": p.get("preaim_angle_sum", 0.0),
                    "preaim_angle_count": p.get("preaim_angle_count", 0),
                    "preaim_angle_mean": (
                        p.get("preaim_angle_sum", 0.0)
                        / max(1, p.get("preaim_angle_count", 0))
                    ),
                    "preaim_error_mean": (
                        p.get("preaim_angle_sum", 0.0)
                        / max(1, p.get("preaim_angle_count", 0))
                    ),
                }
            )

    # チーム別勝率をteam_tacticsに追加
    for team in [team1, team2]:
        t = team_tactics[team]
        # 既に0.0が設定されている場合は上書きしない
        if t["entry_fights"] > 0:
            t["entry_winrate"] = t["entry_wins"] / t["entry_fights"]
        if t["plant_count"] > 0:
            t["post_plant_winrate"] = t["post_plant_wins"] / t["plant_count"]
        if t["retake_attempts"] > 0:
            t["retake_winrate"] = t["retake_wins"] / t["retake_attempts"]
        if t["attacker_rounds"] > 0:
            t["attacker_winrate"] = t["attacker_wins"] / t["attacker_rounds"]
        if t["defender_rounds"] > 0:
            t["defender_winrate"] = t["defender_wins"] / t["defender_rounds"]

    # 最終的なデータを返す（チーム別に完全に分離）
    return {
        "match_metadata": {
            "team1": team1,
            "team2": team2,
            "team1_score": original_data["team1_score"],
            "team2_score": original_data["team2_score"],
            "total_rounds": len(all_rounds),
            "timestamp": "2026-09-22T18:19:16.045087",
        },
        "map_aggregate": {
            # チーム別にメトリクスを格納
            team1: team_tactics[team1],
            team2: team_tactics[team2],
        },
        "round_features": all_rounds,
        "player_stats": player_stats,
    }
