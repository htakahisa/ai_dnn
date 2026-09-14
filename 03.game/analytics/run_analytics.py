#!/usr/bin/env python3
"""
Discitis アナリティクスシステム
試合データ（JSON）を入力し、自動分析レポートを生成する。

使用方法:
    python3 run_analytics.py <json_file_path> [--output <output_file>]

例:
    python3 run_analytics.py series_Ghost_Champions_vs_とうやまゲーミング_0-2_20260913_141446.json
    python3 run_analytics.py series_Ghost_Champions_vs_とうやまゲーミング_0-2_20260913_141446.json --output report.txt
"""

import sys
import argparse
from pathlib import Path
from match_analyzer import load_series_json
from analytics_output import generate_full_report, save_report


def main():
    parser = argparse.ArgumentParser(
        description="Discitis 試合分析レポート生成ツール"
    )
    parser.add_argument(
        "json_file",
        help="SeriesResult JSON ファイルのパス"
    )
    parser.add_argument(
        "--output", "-o",
        help="出力ファイル名（指定なし時はコンソール出力のみ）",
        default=None
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="コンソール出力をスキップ（ファイル出力のみ）"
    )

    args = parser.parse_args()

    json_path = Path(args.json_file)

    # ファイルの存在確認
    if not json_path.exists():
        print(f"❌ ファイルが見つかりません: {json_path}")
        sys.exit(1)

    try:
        print(f"📂 ファイル読み込み中: {json_path}")
        series = load_series_json(str(json_path))
        print(f"✓ 試合データ読み込み完了 ({series.total_maps} マップ)")

        print(f"\n🔄 レーティング計算中...")
        report = generate_full_report(series)
        print(f"✓ レポート生成完了")

        # コンソール出力
        if not args.no_console:
            print("\n" + "=" * 80)
            print("【生成されたレポート】")
            print("=" * 80 + "\n")
            print(report)

        # ファイル出力
        if args.output:
            output_path = args.output
            save_report(series, output_path)
            print(f"\n✓ ファイルに保存しました: {output_path}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
