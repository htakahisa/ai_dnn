"""リアルタイムシーズン専用の、同じ舞台にいる他チームの所属設定。

SEASON_TEAMS にチーム名と players（所属選手名）を設定してください。
先頭5人がロスター、6人目以降が控えです。5人未満のチームも保持できます。
igl と carrier（スパイク所持者）は先頭5人から指定します。
ai は roster_select.py の TEAM_AI_OPTIONS の値（例: default, fnatic_v3,
frc_v1, frc_v1_baseline）で指定します。スクリム準備の初期値に反映します。
省略時はIGLが出場選手の最高IQ、キャリアーが先頭選手、AIがtoru_ai_v3.1です。
transfer_multiplier は移籍金の倍率です（省略時12倍）。引き抜き時の移籍金は
所属選手の基本月給 × transfer_multiplier（円未満切り上げ）になります。
initial_money は内部資金の初期額です（省略時 realtime_season_rival_economy.py の設定）。
debut_chapter は登場章です（省略時1）。設定した章だけに登場します。
選手にも character_stats.py で debut_chapter を指定できます。
initial_rating は初期レートです（省略時1500）。0以上の数値で設定してください。
新規ゲーム・新しく追加したチームに適用し、既存チームの進行中のレートは保持します。
sponsor_active はスポンサー契約の有効・無効です（省略時 True）。
既存チームの資金とスポンサー状態は設定を再読み込みしても保持します。
複数チームに同じ選手を設定した場合、初期所属先をランダムに1チーム決めて保存します。
初期配布で選択した5人は自チームを優先します。同じチーム内での名前重複は不可です。
割当によって5人未満になったチームは、大会で別の出場可能なチームに置き換えます。
competition_manager のプリセットとは独立した設定です。

新しいセーブでは初回起動時に読み込みます。既存セーブへ反映する場合は
python run_realtime_season.py --import-season-teams
で起動してください。既存の自チーム・所持選手・能力値は保持します。
"""

SEASON_TEAMS = [
    {
        "debut_chapter": 1,  # 登場章。この章だけに登場します。
        "name": "Rec1 Gaming",
        "initial_rating": 1550.0,
        "transfer_multiplier": 105.0,
        "players": ["Less", "Aspas", "trent", "keiko", "F0rsakeN", "Verno"],
        "igl": "trent",
        "carrier": "trent",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 1,  # 登場章。この章だけに登場します。
        "name": "Vision Strikers",
        "initial_rating": 1550.0,
        "transfer_multiplier": 105.0,
        "players": ["Mako", "stax", "Buzz", "Rb", "Zest"],
        "igl": "stax",
        "carrier": "Buzz",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 1,  # 登場章。この章だけに登場します。
        "name": "radiantdancer",
        "initial_rating": 1550.0,
        "transfer_multiplier": 105.0,
        "players": ["Sayonara", "Lar0k", "Derke", "marteen", "something", "Loita"],
        "igl": "Sayonara",
        "carrier": "Derke",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 1,  # 登場章。この章だけに登場します。
        "name": "BBL",
        "initial_rating": 1650.0,
        "transfer_multiplier": 75.0,
        "players": ["Rosé", "Lar0k", "Loita", "Crewn", "lovers rock", "CHICHOO"],
        "igl": "Rosé",
        "carrier": "Loita",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 1,  # 登場章。この章だけに登場します。
        "name": "Team Mixed",
        "initial_rating": 1300.0,
        "transfer_multiplier": 75.0,
        "players": ["Brawk", "jawgemo", "Flashback", "Sato", "valyn", "Katarina"],
        "igl": "valyn",
        "carrier": "Brawk",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 1,  # 登場章。この章だけに登場します。
        "name": "Epic Esports",
        "initial_rating": 1350.0,
        "transfer_multiplier": 75.0,
        "players": ["Meiy", "Jinggg", "d4v41", "kaajak", "Wo0t", "vo0kashu"],
        "igl": "kaajak",
        "carrier": "Wo0t",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 1,  # 登場章。この章だけに登場します。
        "name": "Fake Gaming",
        "initial_rating": 1400.0,
        "transfer_multiplier": 75.0,
        "players": ["primmie", "HYUNMIN", "Flashback", "d4v41", "nAts", "IbarakiNinja"],
        "igl": "nAts",
        "carrier": "d4v41",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 2,  # 登場章。この章だけに登場します。
        "name": "Meta Beat",
        "initial_rating": 1300.0,
        "transfer_multiplier": 75.0,
        "players": ["icy", "SiuFatBB", "t3xture", "Asuna", "Zekken"],
        "igl": "SiuFatBB",
        "carrier": "t3xture",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 2,  # 登場章。この章だけに登場します。
        "name": "Team Elites",
        "initial_rating": 1300.0,
        "transfer_multiplier": 75.0,
        "players": ["Tortlilyan", "まーやまくん", "おもこ", "Demon1", "Aspas"],
        "igl": "Tortlilyan",
        "carrier": "おもこ",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 2,  # 登場章。この章だけに登場します。
        "name": "Dragon Moon",
        "initial_rating": 1450.0,
        "transfer_multiplier": 120.0,
        "players": ["Nanasaki", "Canezerra", "WoohyuN", "Zest", "Meteor", "yay"],
        "igl": "Nanasaki",
        "carrier": "Nanasaki",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 2,  # 登場章。この章だけに登場します。
        "name": "Queen's Flower Gambit",
        "initial_rating": 1550.0,
        "transfer_multiplier": 120.0,
        "players": ["leaf", "nAts", "Lar0k", "Chronicle", "Sayonara"],
        "igl": "nAts",
        "carrier": "Sayonara",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 2,  # 登場章。この章だけに登場します。
        "name": "Alien Rex",
        "initial_rating": 1550.0,
        "transfer_multiplier": 120.0,
        "players": ["alecks", "mindfreak", "Wo0t", "something", "eggsterr"],
        "igl": "mindfreak",
        "carrier": "Wo0t",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 2,  # 登場章。この章だけに登場します。
        "name": "Eine Kleine",
        "initial_rating": 1700.0,
        "transfer_multiplier": 180.0,
        "players": ["FNS", "crashies", "cNed", "soulcas", "trexx", "Boaster"],
        "igl": "FNS",
        "carrier": "crashies",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 2,  # 登場章。この章だけに登場します。
        "name": "Japan All-Stars",
        "initial_rating": 1550.0,
        "transfer_multiplier": 100.0,
        "players": ["Laz", "SugarZ3ro", "Meiy", "Dep", "IbarakiNinja"],
        "igl": "SugarZ3ro",
        "carrier": "IbarakiNinja",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 3,  # 登場章。この章だけに登場します。
        "name": "Evil Geniuses",
        "initial_rating": 1600.0,
        "transfer_multiplier": 65.0,
        "players": ["Boostio", "Ethan", "jawgemo", "C0M", "Demon1", "s0m"],
        "igl": "Boostio",
        "carrier": "Ethan",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 3,  # 登場章。この章だけに登場します。
        "name": "Ghost Champions",
        "initial_rating": 1400.0,
        "transfer_multiplier": 150.0,
        "players": ["Xdll", "SyouTa", "Absol", "eKo", "SugarZ3ro"],
        "igl": "SugarZ3ro",
        "carrier": "Absol",
        "ai": "ghost_champions_v1",
    },
    {
        "debut_chapter": 3,  # 登場章。この章だけに登場します。
        "name": "Fnatic",
        "initial_rating": 1750.0,
        "transfer_multiplier": 80.0,
        "players": ["Boaster", "Derke", "Alfajer", "Chronicle", "Leo", "Sayonara"],
        "igl": "Boaster",
        "carrier": "Leo",
        "ai": "fnatic_v3",
    },
    {
        "debut_chapter": 3,  # 登場章。この章だけに登場します。
        "name": "Furina Party",
        "initial_rating": 1800.0,
        "transfer_multiplier": 185.0,
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
        "debut_chapter": 3,  # 登場章。この章だけに登場します。
        "name": "Touyama Gaming",
        "initial_rating": 1550.0,
        "transfer_multiplier": 55.0,
        "players": [
            "Tortlilyan",
            "ろびぃな",
            "いぐるん",
            "えんぺん",
            "夢の街",
            "Homelander",
        ],
        "igl": "えんぺん",
        "carrier": "ろびぃな",
        "ai": "touyama_gaming_v2",
    },
    {
        "debut_chapter": 3,  # 登場章。この章だけに登場します。
        "name": "Omoko Gaming",
        "initial_rating": 1750.0,
        "transfer_multiplier": 95.0,
        "players": [
            "ねこさん",
            "とりさん",
            "おもこ",
            "いぬさん",
            "ひつじさん",
            "Deep",
            "A-Train",
            "Stormfront",
        ],
        "igl": "ひつじさん",
        "carrier": "おもこ",
        "ai": "omoko_gaming_v1",
    },
    {
        "debut_chapter": 3,  # 登場章。この章だけに登場します。
        "name": "Gorigons",
        "initial_rating": 1300.0,
        "transfer_multiplier": 95.0,
        "players": ["ごりまる", "ごんごん", "ごんた", "くんた", "くりまる"],
        "igl": "ごりまる",
        "carrier": "ごんた",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 4,  # 登場章。この章だけに登場します。
        "name": "SUPES",
        "initial_rating": 1700.0,
        "transfer_multiplier": 200.0,
        "players": ["A-Train", "Deep", "Homelander", "Stormfront", "Blacknoir"],
        "igl": "Homelander",
        "carrier": "A-Train",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 4,  # 登場章。この章だけに登場します。
        "name": "Carnal Lust Syndicate",
        "initial_rating": 1450.0,
        "transfer_multiplier": 100.0,
        "players": ["koldamenta", "Rossy", "CHICHOO", "Smoggy", "SereNa"],
        "igl": "SereNa",
        "carrier": "Rossy",
        "ai": "toru_ai_v3.1",
    },
    {
        "debut_chapter": 4,  # 登場章。この章だけに登場します。
        "name": "SereNade",
        "initial_rating": 1900.0,
        "transfer_multiplier": 100.0,
        "players": [
            "Katarina",
            "SereNa",
            "Kr1stal",
            "Foxy9",
            "Furina",
            "S1Mon",
            "Arlecchino",
        ],
        "igl": "Furina",
        "carrier": "Kr1stal",
        "ai": "toru_ai_v3.1",
    },
]
