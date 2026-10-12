# EEG＋fNIRS 疲劳检测框架

这份项目用于先审查代码，再决定数据和实际实验。复制整个 `fatigue_training/` 项目到 Linux 服务器即可使用，运行时不依赖历史 `outputs/` 或 `work/`。当前未执行真实数据训练。

建议依次检查：

1. `fatigue_training/configs/protocol.json`：08 稿的七模型范围、07 稿的模型和训练细节、固定外层五折。
2. `fatigue_training/spec.py`、`protocol.py`：可读的固定字段约束、18/6/6 内层与24/6重训划分、调参同分规则、半向上取整和独立采样流。
3. `data.py`、`normalization.py`：输入格式、共同有效窗、覆盖并集、训练名单隔离和等被试标准化。
4. `models.py`：EEG/fNIRS CNN、独立 Transformer、容量 FFN、拼接头及参数断言。
5. `training.py`、`experiment.py`：早停、重训、完整嵌套调度、证据保存和已完成 fit 复用。
6. `metrics.py`、`tests/test_core.py`：单类指标、逐被试预测、配对 bootstrap、原窗口奇偶诊断和实质性回归检查。

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
