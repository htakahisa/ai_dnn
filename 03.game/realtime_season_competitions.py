"""リアルタイムシーズン専用のカレンダーと大会設定。

START_DATE はゲーム内の日付で、現実の時計とは無関係です。
IN_SEASON_PERIODS は毎年繰り返す月日範囲（両端を含む、年またぎも可）。
TOURNAMENTS の日付は YYYY-MM-DD、prizes は順位:賞金（円）です。
start_date 以降、1日1シリーズずつ進行します。
end_date は終了予定日兼参加登録締切です。参加済み大会は全試合終了時に自動終了し、予定日を超えても継続できます。
maps_to_win は先取マップ数（2ならBO3、3ならBO5）。GFのリセットはありません。
format は double_elimination（4～32チーム）または single_elimination（2～32）。
participation_optional=False は強制参加、allow_player_entry=False は相手のみの大会。
opponent_teams を空にすると、専用所属環境のチームを設定順に必要数選びます。
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
        "end_date": "2026-01-29",
        "visible_from": "2026-01-01",
        "team_count": 4,
        "format": "double_elimination",
        "prizes": {1: 1_000_000, 2: 700_000, 3: 600_000, 4: 500_000},
        "normal_maps_to_win": 1,
        "lower_final_maps_to_win": 2,
        "grand_final_maps_to_win": 2,
        "appearance_conditions": {},
        "participation_optional": True,
        "allow_player_entry": True,
        "opponent_teams": [],
    },
    {
        "id": "kachina_2026",
        "name": "リサ杯 2026",
        "start_date": "2026-03-10",
        "end_date": "2026-04-20",
        "visible_from": "2026-01-01",
        "team_count": 8,
        "format": "double_elimination",
        "prizes": {1: 1_000_000, 2: 700_000, 3: 600_000, 4: 500_000},
        "normal_maps_to_win": 2,
        "lower_final_maps_to_win": 3,
        "grand_final_maps_to_win": 3,
        "appearance_conditions": {},
        "participation_optional": True,
        "allow_player_entry": True,
        "opponent_teams": [],
    },
]
