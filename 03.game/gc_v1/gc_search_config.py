"""Ghost Champions Defender Search configuration.

5～9はGC_ROSTER_ORDERの5人に順番に対応します。

GC_DEFENSE_DEPTH_BIAS_BY_MARKER:
    0.0  = 候補を均等確率で選ぶ
    正数 = スポーンから遠い候補（前目）を選びやすい
    負数 = スポーンから近い候補（引き目）を選びやすい

目安:
    +0.5 / -0.5 : 少しだけ偏る
    +1.0 / -1.0 : はっきり偏る
    +2.0 / -2.0 : かなり偏る
    +3.0 / -3.0 : かなり強く偏る

候補数が2個でも10個でも自動的に確率を計算するため、
map_data_search_gc.py側の5～9の個数を変更しても、この辞書の長さを
変更する必要はありません。
"""

GC_DEFENSE_DEPTH_BIAS_BY_MARKER = {
    5: 1.0,
    6: 0.0,
    7: -1.0,
    8: 1.0,
    9: 1.5,
}
