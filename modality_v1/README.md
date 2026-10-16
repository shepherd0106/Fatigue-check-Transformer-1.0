# 同窗 CNN 单模态对照 v1

新增EEG-only `C_E`与fNIRS-only `C_F`，在相同paired-valid窗口、四个内层折和seed17下，与既有双模态CNN `C_EF`比较。两个新模型分别只有1802和1634个可训练参数；EF为3434。因此结果描述给定训练方案下的表现，不是等容量比较或模态信息上限。

模型从冻结、未训练EF实例初始化：保留活跃CNN分支/全窗均值/LayerNorm32，头复制EF对应32列与bias；不迁移训练后checkpoint。BN1 beta固定零。只加载当前活跃模态的信号、只执行该分支。共同头dropout概率和seed沿用，输入维度不同，不声称随机mask相同。

## 固定设置与预算

原v2共同有效窗、18训练/6验证人、fold2/3/4/5、seed17、lr0.001 Constant；AdamW/原decay分组、eps1e-8、betas(.9,.999)、clip1、batch64、64更新/epoch、最多60epoch、patience10、主指标提高>1e-4。FP32、无AMP/TF32。归一化直接复用相同训练人的旧C工件；完整抽样流重放并核原保存数组及SHA。EF四fit复用，不重新训练。

| 阶段 | 新逻辑run/fit | step上限 |
|---|---:|---:|
| preflight：每模型两次独立两步短训练，另B1/2/64形状检查 | 2 | 8 |
| tiny：共用128窗，两个模型各512更新 | 2 | 1024 |
| research：两个单模态各四fold | 8 | 30720 |
| 合计 | **12** | **31752** |

研究阶段使用真实早停，所以全部完成时实际step可以小于31752。每次step前持久预留额度，失败/中断计账；不自动重试或删除ledger重新获得预算。preflight固定重复容差rtol1e-4/atol1e-5；通过后执行tiny，tiny两模型实际终点accuracy≥.95才自动进入研究。未达marker先保存证据，根据真实曲线复核，不预填通过。

每64更新记录固定训练probe（18人×32窗）与验证逐窗logit、CE、主指标、两类支持/召回及正类预测比例。保留best/last、初始化、分组、采样和best重载。研究最后评估点是实际早停/到上限的位置，区别于上一批强制3840步诊断。只在双方完成的同一fold内比较，四fold齐备后才给完整汇总。

EF学习率来自旧网格，本批不为两个单模态重新搜lr；这是固定共同方案下的对照。旧EF没有logit/CE时CE记NA。单类26号保留，疲劳类召回缺支持记NA。四折是当前开发集内部评价，不是新独立测试或formal验收。

## 路径与启动

- 新代码：`/root/fatigue-modality-v1-code`
- 新结果：`/root/fatigue-modality-v1`
- 新备份：`/root/fatigue-modality-backups`
- 原结果/代码/E0及6run诊断继续原位保留。

上传包需校外部ZIP SHA，并解压到一个尚不存在的新代码目录。启动：

```bash
bash /root/fatigue-modality-v1-code/start_modality.sh
```

流水线后台顺序执行：校交付→完整备份旧dev-v2/E0/代码和已完成的6run诊断（含权重/采样）→metadata prepare→preflight→tiny→研究8fit→report→审查导出。原始信号数据、安装环境和完整Git历史不在训练产物备份中，须继续分别保留。备份应下载或另存后再考虑服务器清理。

查看短进度：

```bash
/root/miniconda3/bin/python -B /root/fatigue-modality-v1-code/check_modality.py
```

日志保存完整输出，结束后给出`/root/fatigue-modality-v1-audit-*.zip`准确路径和SHA；该审查包不含pt/npy，原件留服务器结果目录。任何错误停止流水线并保留已用预算和证据。若进程被强杀留下锁，先检查进程/日志/账本，不直接删锁重跑。

本机只完成元数据/逻辑/代码检查；没有Torch时跳过实际模型/评估状态测试。服务器新preflight、tiny和研究必须以真实结果为准，旧E0不代表新模型已通过GPU测试。时间映射、物理同步和光学转换的科学来源待办不因本批运行而完成。

最终本机独立检查：28项测试26通过、2因缺Torch跳过，含四fold真实元数据prepare（禁止np.load/fingerprint）、仅活动模态访问、账本/早停/部分折比较与备份保护。未运行服务器GPU或研究fit。
