"""章ごとのリーグ名。編集後はリアルタイムシーズンを再起動してください。

選手は character_stats.py の CharacterStats(..., debut_chapter=2)、
チームは realtime_season_teams.py の "debut_chapter": 2 で登場章を設定。
省略すると登場章は1。選手は登場章以降、チームは設定した章だけに登場します。
各章の進行は別セーブ。1章は従来の保存先、2章以降は _chapter番号 を付けます。
LEAGUE_NAMESに4章以降を追加すると、章選択画面にも表示されます。
"""

LEAGUE_NAMES = {
    1: "スタートリーグ",
    2: "ストロングリーグ",
    3: "メイドリーグ",
    4: "モンスターリーグ",
}
