【前提】
・機械学習については、基本的に行動は学習によって対応し、推論側にロジックを入れて行動を制御することは可能な限りせずに、学習させた結果として行動させるようにする。
・ゲームのcore となるファイルはabilities_los.py, battle_logic.py, game_core.py, map_data.py, map_data_defender_setup.py, run_game.py
あたりです。
・基本的には、coreのファイルの修正は不要な認識で、明確な仕様変更、バグがあるときのみ修正対象とします。
・学習用の train ファイルと、推論用の learning ファイルについては、それぞれ対になっているため、例えば、 train_carry.py は learning_carry.py に import してOKです。
ただし、train_carry.py を 別の推論の learning_guard.py 等への import は禁止します。
・学習の実行は絶対に勝手にしないでください。こちらでします。


【説明】
このプロジェクトでは、valorant を上から見たようなゲームを作成しています。
5v5で、それぞれグリッドのマスを1tick に一度移動したり、アビリティ(smoke, recon, flash)を使用できます。
ルールはvalorant と同じで、attacker, defender がいて、スパイクをプラントしたり、解除したりというものです。キャラは8方向を向くことができます。

このゲームをAIに学習させて移動させたり、アビリティを使用したりさせたいです。
射撃は自動で行われるので、AIがするのは、移動とアビリティの使用のみです。

マップのデータは map_data.py にあり、 0が床、1が壁(壁は移動不可能)、2がスパイク配置可能な床、3がattacker の配置位置、4がdefender の配置位置です。


