"""Ghost Champions Defender Search configuration.

map_data_search_gc.py の 5～9 は選手IDではなく、ポジションの
「アグレッシブさ」を表す。

5 = 最も引き目 / 安全寄り
6 = やや引き目
7 = 標準
8 = やや前目
9 = 最も前目 / アグレッシブ

各ラウンド、5段階を5人へランダムに1つずつ割り当てる。
したがって Xdll=5 のような固定対応は存在しない。
"""

GC_SEARCH_AGGRESSION_MARKERS = (5, 6, 7, 8, 9)

# 同じ数字の候補が複数ある場合、最短候補を基本にしつつ少し散らす。
GC_SEARCH_POSITION_RANDOMNESS = 0.35

# 数字が大きいほど、少ない/遠い敵情報でも持ち場を離れやすい。
GC_SEARCH_RELEASE_BY_MARKER = {
    5: {"min_seen": 3, "max_bfs": 6},
    6: {"min_seen": 2, "max_bfs": 8},
    7: {"min_seen": 2, "max_bfs": 10},
    8: {"min_seen": 1, "max_bfs": 14},
    9: {"min_seen": 1, "max_bfs": 999},
}

# 古いimportへの互換性だけ残す。
GC_DEFENSE_DEPTH_BIAS_BY_MARKER = {m: 0.0 for m in GC_SEARCH_AGGRESSION_MARKERS}
