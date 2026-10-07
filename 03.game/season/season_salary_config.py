"""成績連動月給の設定。新規シーズンで選択した時点の値をセーブします。"""

from pathlib import Path


# コンペティション結果だけを集計します。series_data は重複するため含めません。
COMPETITION_RESULTS_DIR = Path(__file__).resolve().parent.parent / "competition_results"
A = 30
MIN_GAMES = 10  # games は出場マップ数。
MAX_MONTHLY_CHANGE = 0.20

# 読み込まない結果ファイル名。元の大会結果は変更・削除しません。
EXCLUDED_RESULT_FILES = (
    "series_Fnatic2023_vs_Omoko_Gaming_2-0_20260927_143901.json",  # JSON構文エラーのため除外
)

# 円単位。固定給は試合数・成績・契約期間の割引・前月比制限より優先します。
FIXED_MONTHLY_SALARIES = {
    # "選手名": 10_000_000,  # 例: 月給1000万円
}
