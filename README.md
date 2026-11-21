# genesis_tools

## 対応環境

- Ubuntu 20.04
- Python 3.10
- Genesis 1.3.3
- rsl-rl-lib 5.5.1
- PyTorch 2.11.0（CUDA 12.8またはCPU向けwheel）

- **学習データの互換性：** 現行版はrsl-rl 5.5.1のactor/critic設定とTensorDictの
  観測グループを使用する．rsl-rl 2.2.4で作成したcheckpointは明示的に拒否するため，
  この環境で再学習するか旧環境で評価する必要がある．
- **PyTorchとの関係：** PyTorchのversionは，rsl-rlだけでなくホストのCUDAおよび
  driver環境に合わせて選択する．GPU向け固定環境はPyTorch 2.11.0とCUDA 12.8の
  組み合わせで検証している．CPUのみの環境ではCPU向けrequirementsを使用する．
  PyTorch 2.8.0はrsl-rl 5.5.1の必須要件ではない．

## インストール

インストール方法は2種類ある．

- **Catkin：** ROS packageをbuildし，`rosrun`のentry pointを使用する．
- **Pythonのみ：** catkin workspaceをbuildまたはsourceせず，`pip`でrepositoryを
  installしてPython moduleのentry pointを使用する．

どちらの方法でも，`genesis_tools`で固定した依存関係のversionを使用する．

### Catkin

1. Python 3.10をinstallする．

   ```bash
   sudo apt install python3.10 python3.10-venv python3.10-dev -y
   ```

2. catkin workspaceに`genesis_tools`をcloneしてbuildする．

   ```bash
   cd <catkin_workspace>/src
   git clone git@github.com:kindsenior/genesis_tools.git
   cd ..
   deactivate  # build前に有効なvirtual environmentを無効化する．
   catkin build genesis_tools
   ```

3. virtual environmentを作成して有効化する．

   ```bash
   mkdir -p ~/genesis_ws
   python3.10 -m venv ~/genesis_ws/venv_genesis
   source ~/genesis_ws/venv_genesis/bin/activate
   ```

4. versionを固定したGenesis 1.3.3環境をinstallする．

   ```bash
   source <catkin_workspace>/devel/setup.bash
   roscd genesis_tools
   python -m pip install --upgrade pip
   python -m pip install -r requirements_Ubuntu20.04_gpu.txt
   # CPUのみの環境では，代わりにrequirements_Ubuntu20.04_cpu.txtを使用する．
   ```

5. installされたversionを確認する．

   ```bash
   python -c 'from importlib.metadata import version; print("genesis-world", version("genesis-world")); print("rsl-rl-lib", version("rsl-rl-lib")); print("torch", version("torch"))'
   ```

   想定するversionは`genesis-world 1.3.3`，`rsl-rl-lib 5.5.1`，および
   requirementsファイルで選択したPyTorch（`2.11.0+cu128`または
   `2.11.0+cpu`）である．

### Pythonのみ

1. Python 3.10をinstallし，virtual environmentを作成する．

   ```bash
   sudo apt install python3.10 python3.10-venv python3.10-dev -y
   mkdir -p ~/genesis_ws
   python3.10 -m venv ~/genesis_ws/venv_genesis
   source ~/genesis_ws/venv_genesis/bin/activate
   ```

2. 任意の作業directoryにrepositoryをcloneする．

   ```bash
   cd <work_directory>
   git clone git@github.com:kindsenior/genesis_tools.git
   ```

3. versionを固定した依存関係とPython packageをinstallする．

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r genesis_tools/requirements_Ubuntu20.04_gpu.txt
   # CPUのみの環境では，代わりにrequirements_Ubuntu20.04_cpu.txtを使用する．
   python -m pip install -e ./genesis_tools
   ```

4. installされたversionを確認する．

   ```bash
   python -c 'from importlib.metadata import version; print("genesis-world", version("genesis-world")); print("rsl-rl-lib", version("rsl-rl-lib")); print("torch", version("torch"))'
   ```

   想定するversionは`genesis-world 1.3.3`，`rsl-rl-lib 5.5.1`，および
   requirementsファイルで選択したPyTorch（`2.11.0+cu128`または
   `2.11.0+cpu`）である．

## サンプル

相対指定したlog pathは，現在の作業directoryを基準に解決される．学習，評価，
exportでは同じlog directoryを指定すること．

### Go2

#### Catkinエントリポイント

```bash
source <catkin_workspace>/devel/setup.bash
source ~/genesis_ws/venv_genesis/bin/activate

# 学習：最終checkpointとしてmodel_100.ptを作成する．
rosrun genesis_tools go2_train.py \
  -l logs/go2_locomotion/test --max-iterations 100

# 評価
rosrun genesis_tools go2_eval.py \
  -l logs/go2_locomotion/test --ckpt 100

# LibTorch向けexport
rosrun genesis_tools go2_export.py \
  -l logs/go2_locomotion/test --ckpt 100
```

#### Pythonモジュールのエントリポイント

```bash
python -m genesis_tools.examples.go2.train \
  -l logs/go2_locomotion/test --max-iterations 100
python -m genesis_tools.examples.go2.eval \
  -l logs/go2_locomotion/test --ckpt 100
python -m genesis_tools.examples.go2.export \
  -l logs/go2_locomotion/test --ckpt 100
```

どちらのexport方法でも，LibTorchから読み込むための`policy_traced.pt`を作成する．
学習，評価，exportでは，標準の設定artifactとして`cfgs.yaml`を使用する．
既存の学習directoryは，`--resume`または明示的な破壊的optionである`--overwrite`を
指定しない限り拒否する．

新規学習で`--max-iterations N`を指定すると，iteration 0からNまで実行するため，
最終checkpointは`model_N.pt`になる．`--stop-iteration N`は代わりに絶対的な
global iterationを指定する．`--max-iterations`との同時指定はできない．

### SBR1

package化されたSBR1サンプルは，Go2と同じ共通rsl-rl境界を使用しながら，
ロボット固有の環境を`genesis_tools.examples.sbr1`に保持する．

```bash
# Catkinエントリポイント
rosrun genesis_tools sbr1_train.py \
  -l logs/sbr1_locomotion/test --max-iterations 1000
rosrun genesis_tools sbr1_eval.py \
  -l logs/sbr1_locomotion/test --ckpt 100
rosrun genesis_tools sbr1_export.py \
  -l logs/sbr1_locomotion/test --ckpt 100

# Pythonモジュールのエントリポイント
python -m genesis_tools.examples.sbr1.train \
  -l logs/sbr1_locomotion/test --max-iterations 1000
python -m genesis_tools.examples.sbr1.eval \
  -l logs/sbr1_locomotion/test --ckpt 100
python -m genesis_tools.examples.sbr1.export \
  -l logs/sbr1_locomotion/test --ckpt 100
```

SBR1も`cfgs.yaml`を書き出し，既存の学習directoryの意図しない置換を拒否する．
checkpointからのresumeに対応し，`policy_traced.pt`をexportする．

### モデルとソースの来歴

`genesis_tools.provenance`は，content-addressed形式のロボットモデルsnapshotと
学習の来歴を記録するための共通helperを提供する．モデルhashは，URDF，URDFが
参照するすべてのファイルresource，およびGenesisのモデルbuild optionを対象に
計算する．snapshotでは，archiveおよびfallback利用のために元の相対pathを
維持する．`inspect_model_source()`と`verify_model_source()`を使用すると，
元のsource modelがarchive済みhashと一致することを確認しながら読み込める．

Gitの来歴には，revision，remote，branch，status，すべての未追跡ファイル名，
および追跡・未追跡のtext fileに対するAIが読めるpatchを記録する．packageの
versionとPython interpreterは，別途`software_environment.yaml`へ書き出す．
`genesis_tutorials`のJAXON3サンプルは，正確なrun pathと共有のcontent-addressed
ロボットモデルstoreとともに，これらのhelperを使用する．

### モーション記録

`genesis_tools.motion_recording.MotionRecorder`は，構造解析用motion bundleを
書き出すロボット非依存のwriterである．ロボット固有のsamplerから数値または文字列の
sampleを受け取り，明示的なreset境界でsequenceを分割し，連続する各sequence内で
関節加速度を導出する．出力する圧縮NPZファイルは，`allow_pickle=False`で
読み込める．

bundle schema 1ではhash付きの`effective_dynamics.yaml`が必須で，各時系列NPZには
sequence schema 1を使用する．2つのversion fieldを分けることで，モデル再構築に
必要なsimulator構築後のlink inertia，joint frame/axis，実行時DOF parameterを
記録しながら，motion array形式だけを独立して更新できる．

ロボット固有のGenesis state/contact抽出，policy読み込み，link選択は
`genesis_tutorials`に残す．最初のadapterはJAXON3である．

### 旧形式pickleの変換

`export_cfgs_to_yaml.py`は，`cfgs.pkl`を含む信頼できる旧形式または上流由来のlogに
だけ使用すること．pickleの読み込みではcodeが実行される可能性があるため，
信頼できないファイルには使用しないこと．

```bash
rosrun genesis_tools export_cfgs_to_yaml.py --log_dir <legacy-log-directory>
# Catkinを使用しない場合：
python -m genesis_tools.legacy.pickle_to_yaml \
  --log_dir <legacy-log-directory>
```
