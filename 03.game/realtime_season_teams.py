"""リアルタイムシーズン専用の、同じ舞台にいる他チームの所属設定。

SEASON_TEAMS にチーム名と players（所属選手名）を設定してください。
1チーム5人以上で、先頭5人がロスター、6人目以降が控えです。
igl と carrier（スパイク所持者）は先頭5人から指定します。
ai は roster_select.py の TEAM_AI_OPTIONS の値（例: default, fnatic_v3,
frc_v1, frc_v1_baseline）で指定します。スクリム準備の初期値に反映します。
省略時はIGLが出場選手の最高IQ、キャリアーが先頭選手、AIがdefaultです。
transfer_multiplier は移籍金の倍率です（省略時12倍）。引き抜き時の移籍金は
所属選手の基本月給 × transfer_multiplier（円未満切り上げ）になります。
同じ選手を複数チーム、または自分の所持選手に重複登録できません。
competition_manager のプリセットとは独立した設定です。

新しいセーブでは初回起動時に読み込みます。既存セーブへ反映する場合は
python run_realtime_season.py --import-season-teams
で起動してください。既存の自チーム・所持選手・能力値は保持します。
"""

SEASON_TEAMS = [
    # 設定例（自分の所持選手と重複しない名前にしてください）
    {
        "name": "Fnatic",
        "transfer_multiplier": 15.0,
        "players": ["Boaster", "Derke", "Alfajer", "Chronicle", "Leo"],
        "igl": "Boaster",
        "carrier": "Leo",
        "ai": "fnatic_v3",
    },
    {
        "name": "Evil Geniuses",
        "transfer_multiplier": 15.0,
        "players": ["Boostio", "Ethan", "jawgemo", "C0M", "Demon1"],
        "igl": "Boostio",
        "carrier": "Ethan",
        "ai": "toru_ai_v3.1",
    },
    {
        "name": "Furina Party",
        "transfer_multiplier": 35.0,
        "players": [
            "Furina",
            "Lohen",
            "Arlecchino",
            "Lisa",
            "Jean",
            "Tartaglia",
            "Kachina",
        ],
        "igl": "Furina",
        "carrier": "Furina",
        "ai": "frc_v1",
    },
    {
        "name": "BBL",
        "transfer_multiplier": 25.0,
        "players": ["Rosé", "Lar0k", "Loita", "Crewn", "lovers rock"],
        "igl": "Rosé",
        "carrier": "Loita",
        "ai": "toru_ai_v3.1",
    },
    {
        "name": "Dragon Moon",
        "transfer_multiplier": 50.0,
        "players": ["Nanasaki", "Canezerra", "WoohyuN", "Zest", "Meteor"],
        "igl": "Nanasaki",
        "carrier": "Nanasaki",
        "ai": "toru_ai_v3.1",
    },
    {
        "name": "Touyama Gaming",
        "transfer_multiplier": 10.0,
        "players": ["Tortlilyan", "ろびぃな", "いぐるん", "えんぺん", "夢の街"],
        "igl": "えんぺん",
        "carrier": "ろびぃな",
        "ai": "touyama_gaming_v2",
    },
    {
        "name": "Omoko Gaming",
        "transfer_multiplier": 10.0,
        "players": ["ねこさん", "とりさん", "おもこ", "いぬさん", "ひつじさん"],
        "igl": "ひつじさん",
        "carrier": "おもこ",
        "ai": "omoko_gaming_v1",
    },
    {
        "name": "Ghost Champions",
        "transfer_multiplier": 20.0,
        "players": ["Xdll", "SyouTa", "Absol", "eKo", "SugarZ3ro"],
        "igl": "SugarZ3ro",
        "carrier": "Absol",
        "ai": "ghost_champions_v1",
    },
]
