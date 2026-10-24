# 個人能力の発動演出

設定ファイルは `data/personal_ability_effects.json`。
キャラ名 → 能力名 → 演出設定の順で記載する。未設定の能力は通常どおり発動する。
現在は、おもこの「追い詰められたおもこ」の発動に接続している。
他の個人能力を接続するときは、その発動境界で
`_queue_personal_ability_effect(char, ability_name)` を呼び、通常tickを進める前に
演出を開始する。

おもこ用の素材は、ACE演出と同じディレクトリへ以下の名前で配置する。

```text
image/ability_omoko01.png
image/ability_omoko02.png
sound/ability_omoko.mp3
```

素材のパスは作業ディレクトリによらず `03.game/` 基準。
素材を配置したらゲームを再起動する。画像ファイルがない場合はキャラ名と
能力名を表示し、音声がない場合は無音で動作する。

設定例：

```json
{
    "おもこ": {
        "追い詰められたおもこ": {
            "frames": ["image/ability_omoko01.png", "image/ability_omoko02.png"],
            "frame_interval_ms": 500,
            "duration_ms": 2000,
            "sound": ["sound/ability_omoko.mp3", 0.5]
        }
    }
}
```

`frames` は表示順の画像一覧、`frame_interval_ms` は画像の切り替え間隔、
`duration_ms` は演出全体の長さ、`sound` は音声パスと音量。
演出を無効にするときは該当設定を削除するか `duration_ms` を0にする。

演出中は移動・射撃・ラウンド時間・スパイク時間・能力の持続時間を進めない。
終了後、発動条件を満たしたtickの処理へ戻り、能力の移動と射撃を開始する。
一時停止・再開も可能。ラウンド／マップのスキップとheadless処理では演出を省略する。
ACEの画像読み込み・中央バナー表示・音声再生の仕組みを共用する。

表示だけの追加で、観測やチェックポイント形式、射線判定、シナリオ識別値の
変更はなく、この演出のための再学習は不要。
