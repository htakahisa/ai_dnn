"""ライバルチームの非公開の資金・移籍判断の設定。金額は円単位。"""

# 新規チーム・資金の記録がない旧セーブに付与する資金。
# SEASON_TEAMS の各チームに initial_money を書くと個別に指定できます。
INITIAL_MONEY = 10_000_000

# 自チームから引き抜く移籍金 = 選手の基本月給 × この倍率（端数切り上げ）。
# ライバル間の移籍には売り手の transfer_multiplier を使います。
PLAYER_TRANSFER_MULTIPLIER = 50.0

# 正規メンバーの復帰とは別に、月に一度、補強オファーを検討する確率。
NON_REGULAR_OFFER_CHANCE = 0.25

# 契約に必要な資金に加え、現在のチームの何か月分の給与を残すか。
OFFER_SURPLUS_MONTHS = 3
