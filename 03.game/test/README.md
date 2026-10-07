# テストの実行

`03.game` 直下にあった `test_*.py` は、このディレクトリに移動しました。
`03.game` を作業ディレクトリにして実行してください。

```powershell
python -m unittest discover -s test -p "test_*.py"
```

特定のファイルだけを実行する場合:

```powershell
python -m unittest discover -s test -p "test_game_tick_time.py"
```
