from __future__ import annotations

from pathlib import Path

TARGET = Path("combo_awakening.py")

OLD = '''        if condition == "enemy_count_at_or_below":
            try:
                maximum_enemies = int(value)
            except (TypeError, ValueError):
                return False

            alive_enemies = sum(
                1
                for candidate in self.chars
                if candidate.team != char.team and candidate.is_alive
            )
            return char.is_alive and alive_enemies <= maximum_enemies

        return False
'''

NEW = '''        if condition == "enemy_count_at_or_below":
            try:
                maximum_enemies = int(value)
            except (TypeError, ValueError):
                return False

            alive_enemies = sum(
                1
                for candidate in self.chars
                if candidate.team != char.team and candidate.is_alive
            )
            return char.is_alive and alive_enemies <= maximum_enemies

        if condition in {"overtime", "ot"}:
            # battle_logic.py が12-12到達時に self.overtime=True にする。
            # OT中のラウンドでは、生存中の覚醒対象に対して成立する。
            return (
                char.is_alive
                and bool(getattr(self, "overtime", False))
            )

        return False
'''


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(
            "combo_awakening.py が見つかりません。"
            "このスクリプトをゲーム本体と同じフォルダで実行してください。"
        )

    text = TARGET.read_text(encoding="utf-8")

    if 'condition in {"overtime", "ot"}' in text:
        print("OT条件はすでに実装されています。")
        return

    if OLD not in text:
        raise RuntimeError(
            "対象箇所が見つかりませんでした。"
            "現在のcombo_awakening.pyの構造が想定と異なります。"
        )

    backup = TARGET.with_suffix(".py.bak")
    backup.write_text(text, encoding="utf-8")

    updated = text.replace(OLD, NEW, 1)
    compile(updated, str(TARGET), "exec")
    TARGET.write_text(updated, encoding="utf-8")

    print("combo_awakening.py にOT覚醒条件を追加しました。")
    print(f"バックアップ: {backup}")


if __name__ == "__main__":
    main()
