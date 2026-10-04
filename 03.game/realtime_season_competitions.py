"""リアルタイムシーズン専用のカレンダーと大会設定。

START_DATE はゲーム内の日付で、現実の時計とは無関係です。
IN_SEASON_PERIODS は毎年繰り返す月日範囲（両端を含む、年またぎも可）。
TOURNAMENTS の日付は YYYY-MM-DD、prizes は順位:賞金（円）です。
start_date 以降、1日1シリーズずつ進行します。
終了予定日は開始日・参加数・大会形式から自動計算します。end_date の設定は不要です。
シングルは参加数-1試合、ダブルは参加数×2-2試合（GFリセットなし）。
参加済み大会は全試合終了時に自動終了し、途中で止めても最後まで続行できます。
maps_to_win は先取マップ数（2ならBO3、3ならBO5）。GFのリセットはありません。
format は double_elimination（4～32チーム）または single_elimination（2～32）。
参加登録は start_date 当日まで。翌日へ進む前に不参加の大会を確認します。
未登録・不参加の大会もライバルのみで自動開催します。
participation_optional=False は事前の不参加ボタンを無効化します（未登録なら当日は不参加）。
allow_player_entry=False は相手のみの大会。
opponent_teams は招待の優先順です。5人未満のチームの代わりにリーグ内の別チームを選びます。
空にすると、専用所属環境のチームを設定順に必要数選びます。
appearance_conditions は全条件を満たすと出現します。対応キー:
min_money, min_owned_players, completed_tournaments（自分が完走した大会IDのリスト）,
best_rank（大会ID:何位以内か）, phase（in_season/off_season）。
visible_from は告知開始日（未指定ならゲーム開始日）。

設定の変更は新規セーブに適用します。既存セーブには
python run_realtime_season.py --import-competitions
で取り込みます。開始済みの大会・日付・契約・所持金は保持します。
"""

START_DATE = "2026-01-01"
IN_SEASON_PERIODS = [("03-01", "11-30")]

TOURNAMENTS = [
    {
        "id": "kachina_2026",
        "name": "カチーナ杯 2026",
        "start_date": "2026-01-15",
        "visible_from": "2026-01-01",
        "team_count": 4,
        "format": "double_elimination",
        "prizes": {1: 10_000_000, 2: 5_000_000, 3: 2_000_000, 4: 1_000_000},
        "normal_maps_to_win": 1,
        "lower_final_maps_to_win": 2,
        "grand_final_maps_to_win": 2,
        "appearance_conditions": {},
        "participation_optional": True,
        "allow_player_entry": True,
        "opponent_teams": [],
    },
    {
        "id": "lisa_2026",
        "name": "リサ杯 2026",
        "start_date": "2026-03-10",
        "visible_from": "2026-02-01",
        "team_count": 4,
        "format": "double_elimination",
        "prizes": {
            1: 50_000_000,
            2: 10_000_000,
            3: 5_000_000,
            4: 2_500_000,
        },
        "normal_maps_to_win": 1,
        "lower_final_maps_to_win": 2,
        "grand_final_maps_to_win": 2,
        "appearance_conditions": {},
        "participation_optional": True,
        "allow_player_entry": True,
        "opponent_teams": [],
    },
    {
        "id": "jean_2026",
        "name": "ジン杯 2026",
        "start_date": "2026-04-20",
        "visible_from": "2026-04-01",
        "team_count": 6,
        "format": "double_elimination",
        "prizes": {
            1: 80_000_000,
            2: 60_000_000,
            3: 30_000_000,
            4: 20_000_000,
            5: 10_000_000,
            6: 5_000_000,
        },
        "normal_maps_to_win": 2,
        "lower_final_maps_to_win": 3,
        "grand_final_maps_to_win": 3,
        "appearance_conditions": {},
        "participation_optional": True,
        "allow_player_entry": True,
        "opponent_teams": [],
    },
    {
        "id": "lohen_2026",
        "name": "ローエン杯 2026",
        "start_date": "2026-06-30",
        "visible_from": "2026-06-01",
        "team_count": 8,
        "format": "double_elimination",
        "prizes": {
            1: 100_000_000,
            2: 80_000_000,
            3: 40_000_000,
            4: 30_000_000,
            5: 20_000_000,
            6: 15_000_000,
            7: 10_000_000,
            8: 7_500_000,
        },
        "normal_maps_to_win": 2,
        "lower_final_maps_to_win": 3,
        "grand_final_maps_to_win": 3,
        "appearance_conditions": {},
        "participation_optional": True,
        "allow_player_entry": True,
        "opponent_teams": [],
    },
    {
        "id": "furina_2026",
        "name": "フリーナ杯 2026",
        "start_date": "2026-09-30",
        "visible_from": "2026-08-01",
        "team_count": 16,
        "format": "double_elimination",
        "prizes": {
            1: 100_000_000,
            2: 100_000_000,
            3: 60_000_000,
            4: 50_000_000,
            5: 40_000_000,
            6: 30_000_000,
            7: 20_000_000,
            8: 10_000_000,
            9: 9_000_000,
            10: 8_000_000,
            11: 7_000_000,
            12: 6_000_000,
            13: 5_000_000,
            14: 4_000_000,
            15: 3_000_000,
            16: 2_000_000,
        },
        "normal_maps_to_win": 2,
        "lower_final_maps_to_win": 3,
        "grand_final_maps_to_win": 3,
        "appearance_conditions": {},
        "participation_optional": True,
        "allow_player_entry": True,
        "opponent_teams": [],
    },
]
