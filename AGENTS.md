# リポジトリ作業規則

## リポジトリ間の役割分担

- ロボットに依存しない再利用可能な共通機能は`genesis_tools`に置く。
- ロボット固有の設定、asset、状態抽出、環境動作は各ロボットサンプル側に置く。
- 共通化によって処理の所在やロボット固有の動作が分かりにくくなる場合は、
  無理に共通化せず各サンプル側に残す。

## 上流実装との対応

- JAXON3
  - 対象：`genesis_tutorials`のJAXON3サンプル
  - 参照元：`start-jsk/jsk_genesis_robots`の`main`
  - 参照範囲：`jaxon3/`
  - 同期状態：`d7df43a65cd1662fde4f7ee0ea9f549793f48cb2`まで比較・反映済み
- Go2
  - 対象：`genesis_tools`のGo2サンプル
  - 参照元：`Genesis-Embodied-AI/Genesis`の`main`
  - 参照範囲：`examples/locomotion/go2_*.py`
  - 同期状態：`v1.3.3`（`76f8f5b3457e7c6d6a078de2244066f9a8694c45`）まで比較・反映済み
- SBR1
  - 対象：`genesis_tools`のSBR1サンプル
  - 参照元：同期対象として設定した上流実装なし
- SampleRobot1
  - 対象：`genesis_tutorials`のSampleRobot1サンプル
  - 参照元：同期対象として設定した上流実装なし
  - 外部モデル：`simple_bipedal` ROS package（repository、branch、commitは未固定）
- 上流を同期するときは、記録済みcommitから新しい参照先までの差分を確認する。
- 各サンプルの上流差分を反映して検証した後、コード変更と同じcommitで
  上記の同期状態を更新する。取り込まない上流差分がある場合は、その理由を明記する。
- 外部モデルやデータを特定の版に固定した場合は、参照元、branchまたはtag、
  commitを同じ一覧に記録する。
- ロボット固有の観測、action、報酬、学習アルゴリズム、環境動作、クラス構成は、
  意図的な差分として明記する場合を除き、参照元の上流実装に合わせる。
- 以下のようなリポジトリ全体に関係する統合機能は維持する。
  - 学習ログの安全な上書き処理
  - YAML設定
  - checkpoint互換性検査
  - provenance
  - ROS entry point
  - Python module entry point
- 上流由来のコードに統合機能を追加する場合は、その差分をプログラム内で
  `# Local integration: <reason>`という形式のコメントを付けて明記する。

## サンプル間の一貫性

- 同等のワークフローについては、ロボット固有の設定や学習タスクに関係しない
  仕様と操作方法をロボットサンプル間で統一する。対象には、コマンドライン
  オプションの名前、alias、意味、既定値、ログパス、resume、上書き防止、
  exportの動作などを含む。
- 共通化できる実装は「リポジトリ間の役割分担」に従って配置する。
- 共通仕様から外れるロボット固有の差分は、プログラム内で
  `# Robot-specific: <reason>`という形式のコメントを付けて明記する。
- 上流実装のあるサンプルをそれぞれの上流実装に合わせた後、その他のサンプルを
  共通インターフェースとリポジトリ規約に合わせる。

## リポジトリ間の文書同期

- `README.md`を編集する場合は、両方のリポジトリが利用可能であれば
  `genesis_tools`と`genesis_tutorials`のREADMEを確認する。
- 両リポジトリで共有する節構成、用語、installation説明、コマンド表記を揃える。
- 共通文書に関係する変更では両方のREADMEを更新する。
- リポジトリ固有の機能、依存関係、制限は維持し、表記を揃えるためだけに
  未対応の機能を記載しない。
- もう一方のリポジトリを確認できない場合は、内容を推測せずその旨を報告する。

## リリースtag

- 依存関係の基準版を示すtagには、次の形式を使用する。
  `genesis-v<Genesis version>_rsl-rl-v<rsl-rl version>[_<purpose>]`
- 同じ依存関係について用途別の基準点が必要な場合だけ`_<purpose>`を付ける。
- tagはannotated tagとして作成する。
- tagを作成する前に必ずREADMEを更新し、tag対象の実装との整合性を確認する。
- 関連する変更をcommitして検証した後にtagを作成する。
- tagをremoteへ明示的にpushし、remote上のtagとcommitを確認する。
