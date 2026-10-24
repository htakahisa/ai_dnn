# チャット操作の試合リプレイ

この試験では、このチャットのAIが公開された観測を読み取り、味方の移動目標を `logs/trial/command.json` に記録します。ゲームは指示されたtick数だけ進み、次の指示まで停止します。既存のゲーム処理が経路探索・アビリティ・射撃・設置を実行します。この試験はOpenAI APIキーを使いません。

このスクリプトは試合前に攻撃側チーム・守備側チーム・守備側AIの選択画面を表示します。攻撃側にはこのチャットのAIが指示を出します。スクリプトはChatGPTモデルの正確なバージョンを取得できません。Start matchを押すと選択画面は進行画面に切り替わり、Setupの経過tick、Battleのtick、スコア、次の指示を待っている状態を表示します。試合画面自体は描画せず、試合は1ラウンドで終了します。試合終了後、このスクリプトは既存の分析画面が読み込める形式で `competition_results/` に試合結果と `replay_frames` を保存します。分析画面で結果を選び、「Replay selected match」を押すとリプレイを再生できます。

作業ディレクトリ：`C:\Users\ronet\MyProject\git\AI_dnn\03.game\chat_play_v1`（このディレクトリへ移動済みの状態）。通常実行コマンド：

```powershell
python cp1_trial_match.py
```

このスクリプトは自動で対戦指示を作りません。チャットのAIが `command.json` に連番・tick数・味方の目標を記録する必要があります。分析画面の起動スクリプトは `03.game/run_replay.py` です。
