# EEG＋fNIRS 疲劳检测框架

这份项目用于先审查代码，再决定数据和实际实验。复制整个 `fatigue_training/` 项目到 Linux 服务器即可使用，运行时不依赖历史 `outputs/` 或 `work/`。当前未执行真实数据训练。

建议依次检查：

1. `fatigue_training/configs/protocol.json`：08 稿的七模型范围、07 稿的模型和训练细节、固定外层五折。
2. `fatigue_training/spec.py`、`protocol.py`：可读的固定字段约束、18/6/6 内层与24/6重训划分、调参同分规则、半向上取整和独立采样流。
3. `data.py`、`normalization.py`：输入格式、共同有效窗、覆盖并集、训练名单隔离和等被试标准化。
4. `models.py`：EEG/fNIRS CNN、独立 Transformer、容量 FFN、拼接头及参数断言。
5. `training.py`、`experiment.py`：早停、重训、完整嵌套调度、证据保存和已完成 fit 复用。
6. `metrics.py`、`tests/test_core.py`：单类指标、逐被试预测、配对 bootstrap、原窗口奇偶诊断和实质性回归检查。

## 直观看训练结果

可视化分为两部分：TensorBoard 在训练过程中查看曲线；离线 HTML 在训练后查看预测和模型比较。两者都不改变模型、训练预算或选模规则。所有命令从包含 `pyproject.toml` 的项目根目录运行。

先看模拟演示，无需 PyTorch 或 TensorBoard：

```bash
python -m fatigue_training visualize-demo --output examples/visualization_demo
```

双击生成的 `examples/visualization_demo/report.html`。这是明确标记的模拟成绩，不是模型训练结果；目录需尚不存在或为空。报告可切换模型、seed、外层折、被试和 run，查看损失/内层验证曲线、混淆矩阵、ROC/PR、每人得分、三个 seed 的模型比较、已有配对 bootstrap 区间，以及逐窗口标签/预测/疲劳概率和错例。图表可下载为 SVG，也可打印为 PDF。HTML 内嵌数据和脚本，无需联网。

真实训练产生结果目录后，生成自己的报告：

```bash
python -m fatigue_training visualize --results runs/experiment01 --output runs/experiment01/report.html
```

可加 `--models M2 M1` 限定显示模型。报告只读日志和预测 CSV，不加载权重或信号数组；中途停止也能生成报告，但会标记当前完成范围。内层验证成绩与外层留出成绩分开显示。逐窗口视图按原窗口索引绘图，缺失窗口断开，不跨 run 连线；概率是类别 1 的输出，暂不解释为经过校准的实际疲劳风险。

在选定的训练服务器环境中安装可选组件，并在原 `train` 命令末尾加 `--tensorboard`：

```bash
python -m pip install ".[monitoring]"
tensorboard --logdir runs/experiment01 --host 127.0.0.1 --port 6006
```

打开训练所在机器的 `http://127.0.0.1:6006`。远程服务器可用 SSH 转发：`ssh -L 6006:127.0.0.1:6006 用户名@服务器地址`，然后在本机浏览器打开该地址。TensorBoard 记录训练损失、内层验证主指标、学习率、耗时和可用的 GPU 峰值内存；外层 refit 不记录逐轮测试成绩。每个 fit/重试使用独立日志目录。

已有 `epochs.jsonl` 日志也能补导出：

```bash
python -m fatigue_training tensorboard-export --results runs/experiment01
```

默认导出到结果目录的 `tensorboard_export`，查看时使用 `tensorboard --logdir runs/experiment01/tensorboard_export --host 127.0.0.1 --port 6006`。可用 `--output` 指定其他导出目录。不开启 TensorBoard 时不要求该依赖；安装包会保留 HTML 模板。当前本机验证没有 PyTorch/TensorBoard，实际事件文件和训练联动仍需在服务器环境验证。

## 服务器环境与入口

Python ≥3.10，NumPy ≥1.26，PyTorch ≥2.6、<3。`requirements.txt` 是兼容范围，服务器上按 CPU/CUDA 驱动选 PyTorch 构建，验证后保存精确版本。网页参考版本不是本机已安装版本。训练代码记录 Python、NumPy、torch、CUDA、cuDNN、设备、配置和源代码哈希。

从包含 `pyproject.toml` 的项目根目录运行所有下述命令：

```bash
cd /path/to/fatigue_training
python -m fatigue_training
python -m fatigue_training plan
python -m fatigue_training plan --models M2 M1 --outer 1 --output runs/plan.json
```

无子命令只显示帮助。`plan` 不读数组、不训练。完整协议为7模型×5外层×27次＝945 fits，最多14,515,200次更新，耗时/显存尚未实测。可以用 `python -m pip install .` 安装为包；配置作为包资源随安装保留。依赖安装仅在选定服务器环境中执行。

## 合成检查与短预算

```bash
python -m unittest discover -s tests -v
python -m fatigue_training synthetic --output runs/synthetic
python -m fatigue_training validate-data --manifest runs/synthetic/manifest.json --output runs/interface.json
python -m fatigue_training smoke-models
python -m fatigue_training smoke-fit --output runs/smoke_fit
python -m fatigue_training overfit-synthetic --max-steps 200
```

前两类数据/指标/协议测试仅需 NumPy。缺少 torch 时，两项 torch unittest 明确 skip；显式 smoke 命令缺 torch 返回错误，不能记为通过。`smoke-models` 检查七模型参数、B=1/2尺寸、CE反向、有限梯度、独立初始化、eval确定性及BN统计不变。`smoke-fit` 只在CPU做合成数据两次更新，验证完整加载/标准化/保存/安全reload/复用链路。过拟合检查使用两条相反标签合成窗口和F0，报告是否达到eval CE<0.1且训练样本全对；它只是记忆能力诊断。

下例仍走完整嵌套调度，但每fit最多两epoch、每epoch一次更新、batch=2，且本次最多新增一个fit：

```bash
python -m fatigue_training train --manifest runs/synthetic/manifest.json \
  --output runs/debug --models M2 --outer 1 --mode interface \
  --max-epochs 2 --updates-per-epoch 1 --batch-size 2 --max-new-fits 1 --execute
```

保持相同命令、数据、配置、环境和输出目录再运行，会复用已完成fit后继续下一fit。每模型每外层仍需16次seed17调参＋8次seed29/43内层＋3次重训，缩短预算不会取消内层验证。完成一个外层的调试链路可将 `--max-new-fits` 设为27。`interface` 模式的合成或未验收真实数据成绩仅用于接口检查，结果标明数据种类、mode及实际预算偏离。

`--config /path/to/custom.json` 位于子命令之前。可调整manifest/路径、相同5×6结构的被试ID分组，以及interface模式中的 batch、updates、max_epochs、patience、improvement_min、bootstrap次数。模型/损失/精度/优化器/采样方法/固定seed/搜索网格等首版约束在 `spec.py` 中逐字段校验，不支持的更改明确报告字段名；新增研究协议应同步修改实现、规格和测试。formal模式另要求原预算64/128/120、patience15、改善1e-4、bootstrap10000。命令行短预算覆盖只能用于interface。

## 数据格式

`manifest.json` 描述来源和每run连续数组；`windows.csv` 描述所有原窗口，包括无效窗。`run_id` 在一个manifest内全局唯一。每run、每模态使用独立文件，禁止多个run（包含不同被试）复用同一个resolve后的信号数组路径。所有NPY都是 `[rows, continuous_samples]`，EEG行数4、fNIRS行数24、dtype=float32；采用只读mmap，转tensor前复制。coverage为bool `[continuous_samples]`。

```json
{
  "schema_version": 1,
  "kind": "real",
  "data_version": "frozen-v1",
  "data_root": ".",
  "window_index": "windows.csv",
  "source": {
    "acquisition": "describe source and hashes",
    "preprocessing": "describe frozen version",
    "label_definition": "stored window F mean >= 0.4"
  },
  "runs": [{
    "run_id": "run001",
    "subject_id": "2",
    "block_info": "block1",
    "eeg": {"array": "run001/eeg.npy", "coverage": "run001/eeg_coverage.npy"},
    "fnirs": {"array": "run001/fnirs.npy", "coverage": "run001/fnirs_coverage.npy"}
  }]
}
```

`kind`必须为`real`或`synthetic`。`source`必须非空，并由数据提供者记录足够的来源、处理版本和标签依据。`data_root`相对manifest目录解析；`--data-root /server/data`可覆盖物理根路径。`window_index`相对manifest目录解析。建议NPY映射均写相对路径；同一manifest和内容可换根目录，物理根路径不参与科学内容指纹。没有coverage文件时，必须显式提供 `"all_samples_covered": true`，这是数据提供者声明，框架仍检查 eligible slices 的长度和有限值。

CSV必要列：

```text
source_index,subject_id,run_id,block_info,window_in_run,eeg_start,eeg_end,fnirs_start,fnirs_end,label_2class,paired_valid,outer_test_fold,invalid_reasons
0,2,run001,block1,0,0,2500,0,250,0,1,1,
1,2,run001,block1,1,1250,3750,125,375,1,0,1,original_missing_fnirs
```

start/end为该run连续轴的半开区间，长度分别2500与250。标签只允许0/1；paired_valid只允许0/1；outer_test_fold必须与配置中的被试分组一致；block_info只能为block1…12或rest1…4。来源索引全局唯一且不重编号；`(run_id,window_in_run)`唯一。无效窗保留完整索引区间及剔除原因，框架不会尝试读取或恢复它。

所有七模型使用同一 `paired_valid=1` 窗口集合。邻窗覆盖拼接或coverage完整均不构成原空窗恢复授权。非重叠诊断直接取原 `window_in_run % 2 == 0` 再筛有效窗；空诊断结果记录 `n=0,evaluation=null`，不会让正常训练失败。

标准化仅拟合train名单。每人每run先合并共同有效窗的连续区间，重叠采样点只计一次；分块float64计算每人均值与中心方差，再等被试合并组内/组间方差。方差floor为该模态正行方差均值×1e-12，记录触及floor的行，无正方差则报错。保存每人样本数、覆盖区间、训练名单和数据指纹；验证/测试只应用训练统计量。内层18人和重训24人分别拟合，缓存绑定数据版本/名单/算法/内容哈希。

## 迁移已有连续数据

```bash
python -m fatigue_training adapt-legacy --manifest /path/to/legacy/manifest.json \
  --window-csv /path/to/07_window_index.csv --data-root /server/continuous/from_10s \
  --output /server/portable_index
```

旧manifest的 `runs.relative_directory` 映射到 `data_eeg.npy`、`data_fnirs.npy` 及对应 `_coverage.npy`。适配器只读元数据，逐字节保留窗口CSV，不访问NPY，不转换PKL，不处理XDF，不改变标签、不建立时间轴验收。只表示源值覆盖的coverage不能证明真实采样同步、无补值、无伪迹或光学转换正确。适配后可用 `--data-root`迁移到另一个服务器根目录，无须保留Windows路径。

## 时间轴 v2 预览

独立入口 `python -m fatigue_training.time_axis` 将已逐样本核对的 legacy processed MAT 信号映射到经过 pyxdf clock synchronization / clock reset handling / dejitter 的时间轴。它不改变训练 CLI、模型或预算。安装预处理依赖可用 `pip install -e '.[preprocessing]'`；固定要求 `pyxdf==1.17.5`，信号执行还需要 SciPy。模块本身不导入 PyTorch，计划校验不导入 SciPy 或 pyxdf。

```bash
python -m fatigue_training.time_axis --source-plan /data/source_plan.json --output /data/preview_v2
python -m fatigue_training.time_axis --source-plan /data/source_plan.json --output /data/preview_v2 --execute
```

第一条命令仅读取 JSON/CSV 元数据并验证来源路径存在，不读取 XDF/MAT 信号，不写输出。第二条命令生成新预览目录；目标已存在即拒绝，包括空目录及中断留下的半成品。重新执行应指定另一新目录。

来源计划 schema 1 的示例如下；路径须为运行环境中的绝对文件路径，SHA256 须来自实际来源。`expected_stream_samples` 必须包含该 XDF 的全部流。`scope_subject_ids` 限定原清单/CSV的被试子集，`runs` 必须逐项映射或明确拒绝该子集中的每个 run。

```json
{
  "schema_version": 1,
  "legacy_manifest": "/data/legacy/manifest.json",
  "legacy_window_csv": "/data/legacy/windows.csv",
  "scope_subject_ids": ["10"],
  "policy": {
    "window_seconds": 10, "stride_seconds": 5,
    "target_rate_hz": {"eeg": 250, "fnirs": 25},
    "max_interpolation_gap_seconds": {"eeg": 0.04, "fnirs": 0.2},
    "label_policy": "preserve_legacy_window_labels"
  },
  "sources": {
    "recording": {
      "path": "/data/raw/recording.xdf",
      "sha256": "replace_with_actual_64_character_lowercase_sha256",
      "expected_stream_samples": {"1": 10000, "2": 1000, "3": 20}
    }
  },
  "runs": [{
    "run_id": "run001", "subject_id": "10", "block_info": "block1",
    "source_id": "recording", "markers_stream_id": 3,
    "start_marker_index": 0, "end_marker_index": 1,
    "expected_marker_start_value": 1, "expected_marker_end_value": 1,
    "marker_raw_time_tolerance_seconds": 0.01,
    "historical_blk": {
      "Task": "NBack", "BlockID": "block1", "start_time": 100.0,
      "end_time": 130.0, "source_xdf": "recording", "KSS": 2
    },
    "eeg": {
      "stream_id": 1, "start_index": 0, "stop_index_exclusive": 7500,
      "raw_mat": "/data/raw/eeg.mat", "raw_variable": "segment",
      "processed_mat": "/data/processed/eeg.mat", "processed_variable": "data"
    },
    "fnirs": {
      "stream_id": 2, "start_index": 0, "stop_index_exclusive": 750,
      "raw_mat": "/data/raw/fnirs.mat", "raw_variable": "fnirs_raw",
      "processed_mat": "/data/processed/fnirs.mat", "processed_variable": "filfnirs"
    }
  }]
}
```

无法追溯的 run 使用 `{"run_id":"run002","subject_id":"10","block_info":"block2","rejection_reason":"source_not_traced"}`。可附 `historical_blk` 等历史信息，保存到 receipt 的完整来源计划；可附 `processed_reference_mat` / `processed_reference_variable`，要求其与 processed MAT 精确一致。MAT 必须为 channels×samples，EEG raw/processed 均4行，fNIRS raw 为16行、legacy processed 为24行；程序不会猜测转置或重新完成光学转换。raw MAT 必须精确等于 XDF 对应 `[start_index,stop_index_exclusive)` 的值，processed MAT 必须有完全相同采样长度。原始 XDF 切片或 raw MAT 存在 NaN/Inf 时拒绝，即使旧 processed MAT 已填补为有限值。程序核对 raw/corrected 中 EEG、FNIRS、Markers 的 type/channel_count/nominal_srate，并要求三类 stream ID 互不相同。每个 XDF 只加载 raw/corrected 两次并缓存，按实际 stream ID 匹配；pyxdf 恢复逻辑导致样本裁剪或值改变时明确拒绝。

每个 run 的公共起点/终点由校正后 marker 指定索引决定，窗口位置固定为 `window_in_run×5` 秒。不同模态使用同一起点的 250/25Hz 连续网格。线性插值只接受精确源采样点或间隔不超过策略阈值的双侧邻点；不能外推、末值填充或跨越大间隔，未支持位置为 NaN / coverage=False。校正时间戳必须有限且严格递增，程序不会排序或修补。侧文件 `*_interpolation.npz` 记录网格时间、左右源 XDF 索引/时间、右侧权重、`direct_source` 与 `interpolation_mask`；coverage 仅表示具有有效数值支持，不能当作原始直接测量标记。

marker 映射还须核对实际 EEG raw MAT 的完整 `blk` 与计划 `historical_blk`。全局 block1–3/4–6/7–9/10–12 分别要求 NBack/SART/MIST/GoNoGo，rest1–4 要求 REST；marker 开始/结束码分别为 1/1、200/202、300/302、400/402、4/4。计划必须给出预期码和不超过0.01秒的 raw 时间容差。raw/corrected marker 值均核验，指定 raw marker index 必须是实际 `blk.start_time/end_time` 的唯一最近点且残差在容差内，report 保存执行时实测残差。来源历史 BlockID 可与全局 stage 不同，合法映射会被报告。pyxdf 警告保留终端输出并分别收集到 raw/corrected 来源报告，附 effective_srate、segments、clock_segments、clock offset、时间/gap 摘要及 file header；`warning_decision=retained_for_nonformal_preview_manual_review_required` 表示待人工核查，不构成科学验收。

输出保留原 CSV 的所有 scoped 行、source_index、被试/run/stage、window_in_run、标签与外层折。原无效窗始终无效；没有可信来源、任一网格点缺支持或窗口结束超过 marker end 时新增排除原因。未追溯 run 仍有 manifest/CSV 行，其连续数组全 NaN、coverage 全 False。输出路径为相对路径，可整体迁移。`report.json` 包含前后窗口资格数量、来源拒绝、完整 source_index 排除列表及时间/索引诊断；`preview_receipt.json` 绑定数据指纹、代码/策略/版本、全部输入哈希和插值侧文件。若全体窗都被排除，预览仍完整保存，report 明确记录现有 WindowDataset 无有效窗口而无法加载。

标签在这里是沿用的每窗口历史监督值，不是对疲劳测量重新插值或重算。receipt 标记 `labels=not_revalidated`，并记录 `legacy_stored_per_window_unverified_after_remap`；物理同步为 `not_established`，光学转换为 `legacy_unverified`，正式验收各项仍 pending。该预览凭据无法通过现有 formal acceptance guard。

## 正式数据与训练

先独立验收时间轴、fNIRS转换和标签生成，再冻结全部输入。`validate-data`检查程序接口、hash、形状/类型/有限值/覆盖及train标准化，不能自动产生这三项研究验收。验收人另存JSON凭据，字段如下（示意值不能作为实际批准）：

```json
{
  "data_fingerprint": "exact SHA256 from validate-data",
  "data_version": "frozen-v1",
  "time_axis": "passed",
  "fnirs_conversion": "passed",
  "labels": "passed",
  "approved_by": "named reviewer",
  "approved_at": "ISO date/time",
  "evidence": "links or paths to audit reports"
}
```

这是核查凭据，不是框架对科学验收内容的自动判断。formal只接受real、三项passed且指纹/版本匹配的凭据，记录凭据内容与hash。合成数据不能进入formal。

```bash
python -m fatigue_training validate-data --manifest /server/index/manifest.json \
  --data-root /server/continuous --output /server/audit/interface.json
python -m fatigue_training train --manifest /server/index/manifest.json \
  --data-root /server/continuous --output /server/results/protocol_v1 \
  --full-protocol --max-new-fits 945 --device cuda:0 --mode formal \
  --acceptance /server/audit/acceptance.json --execute
```

上述945fit命令是可选的完整协议示例，需要评估设备与运行预算后显式执行。单模型/折子集用 `--models` 和 `--outer`；报告注明实际覆盖范围，不能将子集称完整七模型五折实验。真实数据集、实验范围和服务器选择仍待用户审查后确定。

每fit重置Python、NumPy、torch CPU/CUDA随机种子；采样器用seed/外层/内层独立派生的NumPy RNG，不含模型ID或超参，模型初始化/dropout不改变配对窗口抽样流。FP32、无AMP、AdamW全参数衰减、恒定学习率、梯度全局L2裁剪1、无增强。启用deterministic、关闭TF32和cudnn benchmark，CUDA初始化前设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。不支持确定性算子的设备会报错并保存失败状态；同一环境可复现不代表跨平台/版本bitwise一致。

seed17对4网格×4内层主指标均值选型，tie容差1e-12、预定优先序；选定后补seed29/43四内层。每seed四个最佳epoch中位数半向上取整，24人重训重新初始化，不从内层checkpoint接着训练，外层测试只在固定长度重训结束后评价。指标输出argmax，概率相等时类别0；不按概率≥0.5重建标签。

## 结果与恢复

每fit保存 `configuration.json`、`normalization.json`、`epochs.jsonl`、`best.pt`、`predictions.csv`、`result.json`、`status.json`。checkpoint含state_dict、optimizer state_dict及plain metadata，使用 `torch.load(map_location="cpu",weights_only=True)`安全读取。目录已完成且fingerprint匹配时校验全部输出hash并复用；配置、数据、代码、设备或版本改变后拒绝复用，需新输出目录。

失败或中断fit从epoch1重新执行，不支持epoch内或epoch边界续训。重试前把原状态、日志、已有checkpoint等保留到 `attempts/<序号>/`，避免覆盖失败证据。新fit预算只计实际新增fit，`progress.json`记录completed/budget_reached/failed。不同训练进程不能同时写同一输出目录；GPU训练由单一调度者安排。

报告明确区分 `requested_scope_completed`（当前请求范围完成）、`five_outer_folds_completed`（请求模型完成五折）、`full_formal_protocol_completed`（real＋formal＋七模型＋五折＋原协议预算全部完成）。每份报告绑定请求scope和数据/配置hash及生成时间；扩展模型/折范围时先将旧完成报告保存至 `reports/`，当前 `report.json` 改为本次running/budget_reached/failed状态，避免旧结果冒充新计划完成。

数据指纹扫描所有相关NPY字节和CSV并绑定manifest；初始CSV按已解析的同一原始字节hash。缓存指纹前检查文件size/mtime_ns/ctime_ns，训练阶段也检测变化并要求重新载入。stat快照是常见变更检测，不是恶意篡改下的不可变性证明；正式运行必须使用冻结/只读数据。

主指标是每人对真值实际出现的类别宏召回，再被试等权；单类被试AUROC=null。另存固定二类macroF1、两类召回、accuracy、混淆矩阵、可估计AUROC、逐被试与任务/静息分层。最终OOF预测分别保留17/29/43，不选最好seed。主差值M2−M1、F1−F0及次差值M2−E1、F1−F2、M2−M3先平均每人seed差值，再10,000次被试bootstrap（seed20261011）；校验同窗、标签、折、数据指纹和seed集合一致。区间仅描述现有OOF预测的条件不确定性，不涵盖整个调参重训流程。阶段先验只用各外层训练人的16个阶段类别计数，平滑后p==0.5判0，无超参/seed搜索。

参数量：M2=37,626，M1=20,538，F1=18,730，F0=1,642，E1=18,898，F2=18,474，M3=37,370。CNN用BN，Transformer/容量FFN内部及分支池化后用LN；两分支无权重共享、无跨模态attention。范围是离线10秒窗口分类，不能由接口成绩直接证明生理疲劳标签效度、在线因果性或外部泛化。

## 本机验证状态与API依据

本机提供Python3.12.14与NumPy，未装torch。本轮只运行合成/协议测试和无训练CLI检查；torch参数运行计数、前后向、synthetic fit、过拟合、batch64资源试运行均需在服务器执行。实际检查结果记录在 `validation/`（若有）；缺依赖skip与通过分别报告。

最终回归检查共23项：21通过、2因缺少PyTorch跳过；源码语法检查通过。详见 `validation/summary.json`。报告分别标明请求范围完成、五外层完成、完整正式协议完成；预算中断时当前报告反映未完成状态，之前完成的报告保存在 `reports/`。

实现参数及注意事项依据PyTorch官方 [TransformerEncoderLayer](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html)、[Encoder初始化](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)、[AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)、[随机性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)、[torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)。这些引用支持API规则，不代表本机已做PyTorch运行验证。
# 独立开发协议 v2：CNN＋Mamba 分段读出与条件融合

新增实现入口为 `dev-plan / dev-preflight / dev-train / dev-report`，协议ID `cnn_mamba_fusion_development_v2`。保留本文原有v1命令、七旧模型945计划和时间轴代码；新模型不进入旧 `train` 或旧可视化枚举。

当前交付为代码与协议测试。本机没有Torch/Mamba CUDA/sklearn，未执行模型、安装依赖或研究fit；实际5090内核、确定性、完整参数计数及科学验收仍待服务器。PPG/Motion/反向融合/Mamba2/3没有实现。`dev-plan` 显示全部冻结结构/超参与预算，不读取数组、不导入Torch；`dev-preflight` 默认只检查依赖存在性，不写passed凭据。

```bash
python -m fatigue_training dev-plan --phase all --output dev-plan.json
python -m fatigue_training dev-plan --phase D1
python -m fatigue_training dev-preflight --output dev-E0
python -m unittest discover -s tests -v
```

七神经候选C/M0/T/Q/A/F/K可训练参数目标分别3434/43370/37610/51626/46538/54794/54786。两分支 `front.bn1.bias` 保留checkpoint键，但统一固定为初始零、`requires_grad=False`、不进入optimizer；bn1的gamma仍可训练。每模型冻结16个参数，原3450/43386/37626/51642/46554/54810/54802计数作为配置provenance保留。训练态精确算术下，bn1.bias经线性spatial卷积产生的常量位移被bn2居中消去；本修订移除该冗余训练自由度，不使用梯度阈值、梯度掩蔽或更宽比较容差。旧非零bias与既有BN running stats的checkpoint推理不保证等价，不能迁移加载为新模型；新模型明确拒绝非零旧bn1.bias。生产float32、AdamW eps=1e-8、后端和比较容差不变，旧 `models.py` 及945fit协议不变。

四段使用floor边界且互斥，fNIRS为0/12/25/37/50；全窗mean按token等权。新读出投影与调制/容量MLP末层zero初始化。模块构造用独立稳定seed命名空间，各fold/seed共享CNN/head及可共享Mamba初值；每fit实际保存 `initial_base.pt`、组件hash、全部3840×64抽样索引及hash。前端池化、Mamba层与分类头Dropout各自有checkpoint可恢复的独立counter随机流；Transformer整层（含attention dropout）使用独立流。

S_EF保留两层训练人缩放：先用训练被试有效连续覆盖去重后的逐行z-score，再提取104特征，最后以人等权窗口权重拟合feature scaler。fNIRS slope单位为z-score单位/秒。常数信号行与常数特征列scale=1；缓存绑定训练名单、数据与递归代码hash。L2 logistic的sample_weight为 `N_train/(S*n_subject)`，平均1以保留C=1的含义。未收敛停止，迭代单列；为保护中断预算，lbfgs开始前保守预留完整2000迭代额度，日志另存实际n_iter。这可能使失败S槽位没有足够剩余额度重试，不能把额度复位。

依赖extra `development` 声明SciPy/sklearn和精确Mamba包版本，**extra不是已完成环境锁或安装命令授权**。服务器优先保留现有torch2.8+cu128；实施部署时先解析依赖，保存日志、wheel/构建产物与版本，不自动重装CUDA或torch。需要一个clean官方Mamba checkout，HEAD必须为 `10b5d6358f27966f6a40e4bf0baa17a460688128`，安装后整个Mamba Python树须与checkout一致，实际selective_scan扩展文件记录SHA256。预检记录Python、torch build/CUDA、driver、nvcc、CXX11 ABI、compiler、Triton/ninja/einops/packaging/transformers、GPU capability与arch list、causal-conv1d有无与hash。当前严格预检要求这些构建/来源证据；来源不完整即阻断，不凭包版本生成验收结论。

Mamba lazy import仅用官方Mamba-1，无CPU近似或模拟fallback。专用单写入进程第一次导入时固定 `mamba_simple.causal_conv1d_fn/update=None`，目标普通Torch卷积＋selective_scan_cuda，`use_fast_path=False`；causal-conv1d若存在仍记录其来源，但不会参与该专用进程forward。禁止共享线程/研究fit之间切换全局绑定，后续绑定变化立即报错。每窗 `inference_params=None`，无跨窗持久状态。默认float32、禁AMP/TF32、deterministic_algorithms=True；自定义内核是否确定由实测判断。

服务器明确授权后才能执行下列步骤；它们在本机没有执行：

```bash
python -m fatigue_training dev-preflight --execute --device cuda:0 \
  --source-root /root/mamba-locked --output /root/dev-E0
python -m fatigue_training dev-train --execute --phase D0 --max-new-fits 15 \
  --manifest /root/fatigue_time_axis_v2_preview/manifest.json --output /root/dev-v2 \
  --source-root /root/mamba-locked --preflight /root/dev-E0/receipt.json --device cuda:0 --mode interface
python -m fatigue_training dev-train --execute --phase D1 --max-new-fits 60 \
  --manifest /root/fatigue_time_axis_v2_preview/manifest.json --output /root/dev-v2 \
  --source-root /root/mamba-locked --preflight /root/dev-E0/receipt.json --device cuda:0 --mode interface
python -m fatigue_training dev-report --results /root/dev-v2
```

E0完整预检包括七模型B1/2/64前向、独立B64 CE反向/有限梯度与资源测量、每模型两次两step短训练重复/重载（合计28step、7run），再加一次独立CUDA selective_scan与锁定官方selective_scan_ref的前向/梯度数值对照（L50/100，0step）。它不切换研究forward后端。全部合计8run/28step，仍受E0总12run/64step硬上限含失败/重试约束。零末层上游首步零梯度是允许的；第二步检查学习链打开。rtol1e-4/atol1e-5超限即停，不自动放宽。资源数字只来自实际执行，不能当速度优势。

D0必须15槽位全部通过才运行D1。研究D1=60fit、D2=56fit，研究112神经＋4S=116fit、430080神经step；含D0为131槽位、430108step；含E0为430172step。每神经研究fit最多60epoch×64step、batch64，patience10。持久ledger在每optimizer.step前预留额度，失败/中断照计；每fit最多原attempt＋一次显式修复retry，阶段硬上限不翻倍。保留attempt目录与失败日志，不覆盖旧证据；独占writer.lock防并发写。进程硬中断遗留lock必须先核进程/证据再人工恢复，不能删除ledger重置预算。

开发仅原外层1的24训练人，四组18/6内层；六人 `[2,9,12,20,29,33]` 在批次/切片/标准化/采样/预测前由guard隔离。只支持冻结v2真实数据指纹；D0合成只用于接入。interface不是formal验收；`--mode formal`另要求现有数据科学验收receipt且不改变inner scope。

D1选中lr后，若旧C/M0/T三者四折主指标均≤.52，或F在≥3折疲劳支持验证人等权召回≤.10，停D2；任一折疲劳支持缺失记NA并阻断。质量检查通过、曲线/个体偏置审阅后，D2需单独命令 `--phase D2` 及 `--quality-review`：JSON字段 `approved_by/approved_at/curves_and_bias_reviewed=true/engineering_unblocked=true/D1_report_sha256/code_hash`，hash须匹配完整固定D1证据快照 `D1_review_report.json` 与当前代码。这个快照不随scheduler状态/D2新增结果改变，源码或D1证据修订时旧快照另存review_history。D2使用seed17选定lr并只增加29/43，不新选结构。程序不会自动扩展外层重训。

合法修复须在 `--repair` JSON记录 `slot/reason/approved_by/approved_at/old_code_hash/new_code_hash/reviewed_unaffected_slots`，新旧hash不同，slot是失败/中断槽位。只有明确逐槽审阅的不受影响已完成结果能保留旧binding，报告显示实现hash异质性；未审阅则阻断，不能无记录混用。先重新取得当前代码的E0 receipt。E0修复同样 `--repair`，兼容单 `slot` 并支持明确 `slots` 列表，用 `reviewed_unaffected_e0_models`（七模型ID及可选scan_reference）逐项声明，保留旧check/hash；失败check重跑计入12run/64step，不能另开免费重试。修复receipt模板只是人工审阅记录结构，不提供自动批准。

本次bn1修复的独立交付包含 `apply_bn1_repair.py`、原32文件清单、审核后的新payload清单及repair review。脚本仅接受原源码hash `fb01431dbea3e99dd55b87f23d924d0754d0854aa3d52fac294828edce24c900`、原配置、同一环境、C_EF完成/M0_EF重复比较失败、E0累计8step/2run且D0/D1/D2无fit的已观察状态；指定development产物目录也须无fit。全部原attempt1文件原位保留，并在 `history/bn1_zero_frozen_v1/original` 逐字节留档ledger、C结果/status及失败证据，原替换源码另存source。迁移只更新配置binding并追加记录，两旧slot置superseded而保留attempts，计数不变。使用多slot repair使C/M0从attempt2重检，其余五模型及scan reference全部重新检查；成功时累计10run/36step，失败仍消耗现有12run/64step预算。重复运行patch核验既有历史及已推进账本，不重置计数。每次完整重复比较先保存 `repeat_comparison.json`，失败时可检查各tensor、logits和梯度摘要。

修复代码的本机metadata验证与服务器GPU验证须分别报告；冻结策略尚不表示新版E0已通过，不启动研究fit。主agent审核生成精确payload hash和repair授权记录后，按修复交付说明运行脚本，再运行原 `dev-preflight --execute` 命令并追加 `--repair /root/dev-E0/bn1_repair_review.json`。

独立报告 `development_report.json / development_summary.csv / development_report.html` 完全离线，含选定lr、支持数/按人混淆、曲线、阶段/恒定诊断、预算与门、条件bootstrap及探索性交互差。统一 `scope=inner_validation/outer_test_evaluated=false/formal_protocol_completed=false`，不调用旧outer-OOF可视化。HTML只渲染已有预测证据，不能代替模型/GPU/数据科学验收。


## 2026-10-15 首批训练诊断入口

新增独立目录 [diagnostics_v1](diagnostics_v1/README.md)，含已审原模型的冻结副本、独立训练诊断入口与备份/启动/短进度脚本。原fatigue_training的25个py/json内容保持857a9f7b…d4fb；旧116开发fit已完成并审计，尚无稳定融合收益，当前诊断另立6run/16384step上限。诊断本机仅元数据/逻辑测试，GPU实测待服务器，不代表科学/独立/正式验收。

服务器请将单独交付包解压到 `/root/fatigue-diagnostics-v1-code`，产物写 `/root/fatigue-diagnostics-v1`。先完整备份旧结果/E0/代码，再运行两次小集检查；达到预设marker才自动继续四个fold2/seed17的Constant/Cosine诊断。详细步骤以本次工作区交付说明为准，保留本页上述历史阶段说明。
## 2026-10-15 同窗CNN单模态对照入口

新增独立目录 [modality_v1](modality_v1/README.md)：C_E EEG-only / C_F fNIRS-only，从未训练EF实例的共同分支与头列初始化；原25个模型源码及diagnostics_v1保留。四内折seed17、lr.001 Constant/原早停，新增8研究fit、复用旧4个C_EF，另2preflight与2tiny，合计12逻辑run/最多31752step。不是等容量或各自最优超参比较。

本机最终28项测试26通过、2缺Torch跳过；未运行新GPU/研究fit。服务器新代码 `/root/fatigue-modality-v1-code`、新产物 `/root/fatigue-modality-v1`，先保全旧dev-v2/E0和已完成诊断，再按preflight→tiny→research运行。历史D1/D2与诊断结果不重置；科学/独立/formal状态不变。