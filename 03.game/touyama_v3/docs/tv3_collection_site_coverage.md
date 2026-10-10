# 左右のケース収集（2026-10-10）

## 原因と変更

FnaticとGCにはTouyama編成に対する攻撃先の制限がある。ConConは採用経路の選別、Omokoは経路成績、ToruとTouyamaのplantは分析値による選択で片側に偏る。seedを増やしても、選択対象から外れたサイトのデータは得られない。

retake収集の「0件の側をスキップして完了」は廃止した。guard収集は左右合計50件から、相手・サイトごとに50件（左右合計100件）へ変更した。

両収集スクリプトの通常設定は `targeted`。件数の少ないサイトをブロックごとに指定する。指定は収集プロセス内のサイト・経路選択だけに適用し、通常のゲームや評価には適用しない。地形、学習済み重み、射線、HP、初期配置、設置・戦闘ルールは同じ。移動・戦闘から実際に設置が完了した状態だけを保存する。サイト指定ケースは通常の攻撃先分布を再現するデータではなく、左右を学ぶための条件付きケースである。

Fnaticは登録設置マス、GCは既存のサイト指定・戦略選択、ConConは指定側の既存モデル経路、Omokoは指定側の既存経由点、FRCは既存ナビゲーションのサイト指定を使う。Toru/Touyamaは指定側の候補経路を分析して選ぶ。保存ケースとブロック記録に `requested_site`、manifestに `sampling_version` を記録する。通常選択で収集する場合は冒頭定数を `natural` にする。

指定しても死亡・時間切れで設置できないラウンドはある。上限までに件数を満たさなければ、不足として終了コード2を返す。0件を完了扱いにしない。再開時のブロック上限は累計のため、上限到達後に続けるなら上限も増やす。

## 実行

作業ディレクトリは `03.game/touyama_v3/`。設定は各スクリプト冒頭で変更する。

- retake: `DEFAULT_OPPONENTS`、`DEFAULT_SITES`、`DEFAULT_CASES_PER_SITE`、`DEFAULT_SITE_SAMPLING`、`DEFAULT_MAX_BLOCKS`、`DEFAULT_RESUME`。
- guard: `TARGET_OPPONENTS`、`TARGET_SITES`、`CASES_PER_SITE`、`SITE_SAMPLING`、`MAX_COLLECTION_BLOCKS`、`RESUME_COLLECTION`。

```powershell
Set-Location C:\Users\ronet\MyProject\git\AI_dnn\03.game\touyama_v3
python tv3_collect_defender_retake.py
python tv3_collect_attacker_guard.py
```

既存データは今回の修正作業で更新していない。通常の新規収集設定（resume=False）で実行すると、指定出力先の従来ケースを置換する。従来データを保持する場合は、出力先定数を別ディレクトリへ変更する。旧方式のmanifestと新方式のmanifestは再開混在を拒否する。新方式で中断した収集の追加は、同じ設定でresume=Trueにする。

収集後、ユーザーが `tv3_train_defender_retake.py` / `tv3_train_attacker_guard.py` を新規学習する。analysis・search・plantの再学習は今回の収集変更だけを理由に必要にならない。元モデルを更新した場合は依存hashが変わるためケースを再収集する。

retakeは通常対戦で片側の評価が0件だと、その側のbestが保存されない。新しいtargetedケースで学習する場合は、通常対戦評価を別に表示し、独立seedのサイト指定対戦で左右bestを選定する。`SITE_EVALUATION_SEED_COUNT` が指定評価のseed数。保存モデルに指定評価条件を記録する。通常対戦で0件の側について通常分布での勝率を主張しない。既存latestへ新データ・新評価条件を混ぜず、新規学習する。

## 検証

重み更新なしで、6相手すべてについてretakeの不足側の実設置を確認。guardは6相手×左右の実設置ケースを取得し、一時保存・復元を確認した。これらは全目標件数の本収集や学習性能の検証ではない。
