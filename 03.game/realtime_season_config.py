"""リアルタイムシーズンの初期キャラ候補設定。

INITIAL_OWNED_PLAYERS に、character_stats.py に登録されている名前を
1人ずつ文字列で5人以上追加してください。新規ゲームでここから5人を選びます。
選んだ5人だけを入手し、選ばなかった候補はスカウトで契約できます。
候補の重複、存在しない名前、他チーム（控えを含む）の所属選手は設定できません。
保存済みの候補・選択・所持選手・能力値・ロスターは、ここを変更しても上書きしません。
既存セーブは所持選手を保持し、初期選択をやり直しません。
"""

INITIAL_OWNED_PLAYERS = [
    # 設定例（使う選手の行の先頭にある # を外してください）:
    "Less",
    "F0rsakeN",
    "Jinggg",
    "d4v41",
    "Sato",
    "valyn",
    "kaajak",
    "primmie",
    "まーやまくん",
    "Brawk",
    "Smoggy",
    "Lysoar",
    "stax",
    "Mako",
    "Rb",
    "Buzz",
    "icy",
    "SiuFatBB",
    "t3xture",
    "Asuna",
    "Zekken",
]
