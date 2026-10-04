"""ペア練度の設定。リアルタイムシーズンだけに適用します。"""

pair_familiarity_enabled = True
TAU_DAYS = 180
M0 = 0.80
M_MAX = 1.10
# 各段階の開始値。境界値は上の段階に含みます。
STAGES = (("初対面", 0.0), ("顔なじみ", 0.15), ("相棒", 0.40),
          ("長年の相棒", 0.65), ("名コンビ", 0.90))
INITIAL_SEED = 2026
AI_INITIAL_MIN_DAYS = 200
AI_INITIAL_MAX_DAYS = 600
AI_PAIR_VARIATION = 0.20
MAX_NEWS_PER_DAY = 3
PARTNER_LIMIT = 5
RANKING_LIMIT = 20
STAGE_COLORS = ("#f1f5f9", "#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8")

