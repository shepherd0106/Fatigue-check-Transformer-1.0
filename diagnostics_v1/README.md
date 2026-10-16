# 首批训练诊断 v1

用于判断当前训练是否充分，以及固定预算下学习率下降是否值得继续研究。范围为已有开发集的内层 fold 2、seed 17，C_EF 和 M0_EF；外层测试六人的信号不进入训练、probe、预测或统计拟合。强指纹会对冻结数据源文件做字节哈希，属于来源校验。

## 运行内容

| 阶段 | 设置 | 预算 |
|---|---|---|
| 旧产物备份 | 完整 dev-v2、dev-E0，含权重/抽样/特征数组和原代码；核引用文件及 ZIP 内容哈希 | 0 更新 |
| 小集检查 | 按数字编号取前4个双类支持训练人，各16窗/类，共128窗；C/M0各512更新 | 2 run，1024 更新 |
| 单折诊断 | C/M0各 Constant 与 Cosine，均3840更新 | 4 fit，15360 更新 |

首批总上限6个 run/fit、16384次更新，失败前已预留的更新仍计账。已有失败或中断槽不自动重试；不得删除 ledger、writer.lock 或已有结果后重跑。小集终点 accuracy 达到95%才自动进入后续诊断；未达时停止并导出证据，依据实际曲线复核后再决定。

保持原模型、初始化分组、训练人、归一化、优化器、FP32、关闭 TF32、Mamba 后端与 BN1 零偏置策略。Pilot 重放旧实验完整抽样流；Constant 为0.001，Cosine 在更新前从0.001降至0.0001、无 warmup。每64更新评估固定576窗训练 probe 和同折验证集，记录逐窗logit、两类召回/缺类NA、主指标、正类预测比例、窗口等权与被试等权CE；保留 best/last、旧patience10的虚拟截点和固定末段3648/3712/3776/3840。

实际早停关闭，在同一优化器轨迹上继续。更多评估点会增加选择best的机会；判断时比较虚拟截点、固定终点及预设末段，不只比较最高分。小集是工程检查，probe不是新验证集，单折结果不是独立确认。

## 服务器目录

- 新代码：`/root/fatigue-diagnostics-v1-code`（本包）
- 新产物：`/root/fatigue-diagnostics-v1`
- 备份：`/root/fatigue-diagnostic-backups`
- 原代码/研究结果/E0：`/root/Fatigue-check-Mamba-v2`、`/root/dev-v2`、`/root/dev-E0`。

本包带25个原模型冻结文件，内容指纹仍为 `857a9f7b83b4e546be4b73a1e1a0c02370178b1957a4f098674cb9925b50d4fb`。独立wrapper/protocol及源码快照另行绑定，不覆盖原研究目录或预算。备份不含真实信号数据包与已安装Python/CUDA环境，这些需继续分别保留；备份仍位于同一服务器，应下载/另存备份ZIP与回执后再清理服务器。

解压并核验后运行 `bash /root/fatigue-diagnostics-v1-code/start_diagnostics.sh`。启动器先校交付哈希，然后后台顺序备份→prepare→tiny→pilot，打印PID和完整日志位置；所有错误停止流水线。没有远程自动执行，当前交付不表示新wrapper已通过GPU检查。

短进度：

```bash
/root/miniconda3/bin/python -B /root/fatigue-diagnostics-v1-code/check_diagnostics.py
```

结束或正常异常后，日志会给出 `/root/fatigue-diagnostics-v1-audit-*.zip` 的准确路径及SHA。该审查包含JSON/CSV/源码和哈希索引，不含权重/采样数组；其原件留在新产物目录。若强制终止进程而未导出，先保留日志/锁并反馈，不直接删除锁或重新跑。

## 已完成的本机检查及限制

诊断17项测试16通过、1因本机缺Torch跳过；备份10项测试通过。真实元数据prepare在禁止`np.load`和强指纹扫描时成功，tiny128/probe576、0更新、重复结果一致。壳脚本通过语法检查。上述不覆盖CUDA前后向、Mamba新训练轨迹或评估状态的GPU实测；服务器小集将实际检查数值、梯度、BN1、缓冲区及RNG不变。

本批不自动扩其余内折、seed、模态或Mamba2。标签重映射、物理同步和光学转换的科学来源核查仍待；完整来源分析在项目工作区的`outputs/训练诊断实施_20261015/来源核查`。
