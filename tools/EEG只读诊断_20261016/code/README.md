# EEG 只读诊断 v1

本脚本解释 tiny128 与 18 人训练 probe 的差距。固定 fold2 / C_E，分别加载 tiny、research 的 best/last 共四份原权重，在同一 tiny128 与原训练 probe576 上做标准 eval。probe 同时拆为 tiny 四人、其余十四名训练人。它们均是训练人，不能解释为独立泛化或正式验收。研究 best 选择仍来自原验证记录；本轮不产生新验证预测。

新增 fit=0、optimizer steps=0。不构造优化器，不 backward，不切 train，不修改 BN，不重算 normalization，不安装依赖。模型是原 CNN，无需运行 Mamba kernel 或新建 Mamba 训练环境。需要原 CUDA/PyTorch/NumPy 环境，CUDA 缺失即退出。以服务器实跑报告为准；本机 stdlib 测试不能证明 GPU 重现。

## 服务器运行

把本目录五个文件放入新目录 `/root/fatigue-eeg-readonly-v1-code`。不要覆盖旧代码目录，不执行审查 ZIP 内的旧脚本。仍使用现有模态研究环境与原结果目录：

```bash
cd /root/fatigue-eeg-readonly-v1-code
/root/miniconda3/bin/python -B -m unittest -v test_eeg_readonly.py
bash run_readonly.sh
```

默认原代码 `/root/fatigue-modality-v1-code`、原结果 `/root/fatigue-modality-v1`、新输出 `/root/fatigue-eeg-readonly-v1`、device `cuda:0`。入口脚本固定使用 `/root/miniconda3/bin/python` 前台运行，将完整 stdout/stderr 保存 `/root/fatigue-eeg-readonly-v1.log`，拒绝已有输出或日志；成功只展示摘要，失败展示最后25行并保留退出码。CNN 不要求 CUDA_HOME 或 nvcc 检查。`--manifest --baseline --preflight` 可省略，从 prepared binding 解析；显式提供时必须为原绑定绝对路径，不能借机迁移或替换输入。输出已存在就拒绝；再次诊断应使用 `/root/miniconda3/bin/python -B -u eeg_readonly.py --output 新绝对路径` 指定另一个全新、无重叠的输出并保存新日志。不要删除或重置旧结果目录。输出不能位于任何原代码、结果、数据、基线或历史证据目录内，也不能是这些目录的祖先。拒绝 symlink/reparse 路径。

成功或输出目录创建后的失败会自动生成新输出目录内 `eeg_readonly_audit.zip`，上传此 ZIP 与上述启动日志即可。启动前的来源/绑定/CUDA失败仅有终端日志，没有假成功报告。ZIP 只含本次 JSON/CSV/摘要与 SHA 索引，不打包信号、权重、旧数组。不要只上传 summary：详细 JSON 才包含覆盖和复现校验。

## 校验及输出

- `trusted_sources.json` 固定 32 份已审源文件的 SHA；导入原代码前核验。原 `source_snapshot` 仅作字节证据，不执行。必须有原 12/12 completed ledger；先核所有 status/spec/binding/step 与 artifact SHA，再以 `torch.load(weights_only=True)` 加载四份 EEG 权重。旧状态文件自身、关键输入元数据、源码均前后固定 SHA，结束再核全部原 artifact。不会调用原训练/prepare/run/Budget 构造函数。
- 标准 eval 精度沿用原 `configure_precision`：float32、TF32 关、cudnn benchmark 关、deterministic algorithms 开。tiny best 对原 best_reloaded、tiny last 对原终点 probe；research best/last 对各自 step 的原训练 probe。CSV 元数据/顺序必须完全一致，logits 固定 `abs(new-old) <= 1e-5 + 1e-4*abs(old)`，不得调宽。
- 只触及 fold2 tiny/probe 所在训练 run 的 EEG 数组和 coverage mask。先核数据根 `checksums.json` 固定 SHA `e226148b0fbf36e80a733a3d39d49d0410e547687d8dd75caa144d2f09adde8f`，再按清单核这些文件的大小/SHA，结束重核。哈希分块读取对应文件字节；数值分析 mmap 只读按窗口切片。不会读外六人、验证人信号或 fNIRS 信号。元数据可以包含完整队列，但不调用全队列 fingerprint。
- `sampling.json` 核 tiny 两个 fit 和 research 八个 fit 保存的 sampling npy SHA/dtype/shape，mmap 只读，只计 `[:executed_steps]`。research 的 3840 行未执行 suffix 不算曝光。唯一窗口覆盖率以该 fit 的合法训练窗口为分母；tiny 另给全部18训练人分母。曝光 min/median/max/zero 是固定 probe 每窗口被训练抽中的次数，含零。按人记录窗口数、唯一覆盖、draw 次数，另记标签抽样次数。原流按人等权、窗内均匀，不能误称类平衡。
- `inputs.json` 每窗口/四通道，给 raw 和 normalized mean、总体 std（ddof=0）、RMS=`sqrt(mean(x²))`、maxabs、flat（全2500点均有限且精确峰峰值=0）、非有限点数。normalized 使用原18人 normalization，经过与 active_batch 相同的 float32 舍入后汇总。按人/通道记录上述窗口描述量的 min/median/max；不是连续记录统计，重叠样本按每个所选窗口重复计入。非有限值只保留描述，立即失败，不剔除或修复数据。
- 四份 checkpoint JSON 含 BN running_mean/running_var/num_batches_tracked 摘要和各组指标，eval 的 buffer/RNG 不变由原已审 evaluate_fixed 检查。没有 BN 重校准或 train 模式前向。

`report.json` 只有全部复现与前后字节校验通过才标 completed。失败时检查 `failure.json` 与详细日志，不把部分输出当成功。旧权重在服务器保留，本轮不能解决标签物理同步或独立测试证据问题。

## 本机测试边界

`python -B -m unittest -v test_eeg_readonly.py` 仅依赖标准库，覆盖输出/逃逸路径保护、拒绝覆盖、执行 prefix、probe 零曝光、覆盖分母、logit 容差和元数据失败、非有限旧 logits、按训练人限定的 checksum/篡改检测、失败 audit ZIP，以及 AST 的零训练调用与 weights_only 加载约束。测试不会运行模型、读取真实信号或声称 CUDA 已通过。
