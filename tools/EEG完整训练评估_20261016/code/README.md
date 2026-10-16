# EEG fold2完整训练范围的零训练评估

目的：旧probe576只覆盖block1/10/11/12，先评估18人全部11687训练窗，再决定是否需要新fit。本包只加载tiny best、research best、research last三个已完成检查点；没有训练、反向传播、优化器、权重保存、归一化重估或BN重校准。

部署本目录到 `/root/fatigue-eeg-fulltrain-v1-code`，保留既有目录：

- 已审只读诊断代码 `/root/fatigue-eeg-readonly-v1-code`，其中eeg_readonly.py和trusted_sources.json分别固定SHA为 `514cda892d879904dad575875f5f7e9061d6cd04a7d554443b28cf4aacf25944` 和 `62201155217d9c5acddd197091c76f80d424fb4c8a988d84078ec238fbca5257`，导入前校验。
- 原模态代码 `/root/fatigue-modality-v1-code`、结果 `/root/fatigue-modality-v1`、只读诊断结果 `/root/fatigue-eeg-readonly-v1`。后者固定已审audit_index SHA `2ce7ad49d17a04536d9db3449f25b07656671645137dc718d406be71eab36337`，19项证据逐项校验。
- manifest/baseline/preflight由原prepared绑定恢复，CLI给出时必须和绑定完全一致。`--previous-code`、`--previous-results`可指定搬迁后的同内容路径；代码及证据hash仍固定。

服务器执行：

```bash
bash /root/fatigue-eeg-fulltrain-v1-code/run.sh
```

使用 `/root/miniconda3/bin/python`、固定cuda:0，完整数据按原 `dataset.indices(train)` 顺序、batch64，经原evaluate_fixed/no_grad推理，检查buffers与RNG不变。配置精度复用原configure_precision。没有CPU回退。

新输出 `/root/fatigue-eeg-fulltrain-v1`，新日志 `/root/fatigue-eeg-fulltrain-v1.log`；现有输出或日志拒绝覆盖。自定义输出需直接运行Python CLI并另选新日志；新输出禁止与原code/results/data/baseline/preflight、只读诊断code/results及本包code重叠。任何旧工件不做修改。启动时原代码/绑定核验失败不创建输出；输出初始化后的失败（含新源码交付清单不匹配）保留failure.json与partial audit ZIP，run.sh打印最后25行并保留退出码。

全fold2训练人EEG及coverage按已固定checksums校验并前后重核，不加载fNIRS信号、不做validation或外层推理。metadata、status、所有已完成run的artifact inventory以及已审只读诊断证据前后核验。

每个检查点产生full_train逐窗CSV，以及分组JSON/CSV：完整11687、tiny128、tiny4人其余2330窗、其余14人全部9229训练窗、probe576、非probe11111。每组同时给出整体、18人、16任务段、18人×16任务段统计；任务段为block1–block12和rest1–rest4。CSV统计行包含n、两类support、accuracy、primary、窗口等权/按人等权CE、按人等权/池化召回、预测疲劳占比。全CSV加training_universe.json保留精确分组与顺序，分组之间有交叉；tiny128/tiny4其余/other14构成互斥全集，probe/nonprobe构成另一互斥全集。

primary沿用原定义：每人对有支持类别的召回求平均，再对当前有窗人员等权。不存在的类别召回为NA，空组accuracy/primary/CE/召回等为NA；JSON使用null，CSV使用NA。单类block的primary按其支持类计算，不应与双类平衡指标混淆。tiny4人其余窗口不是独立测试；其余14人仍是研究训练人，整个结果是已观察开发数据的内部诊断。

完整前向结果按source_index映射抽取原tiny/probe顺序，与已审只读诊断CSV以及原研究/小集CSV逐行核对元数据、logits固定`rtol=1e-4, atol=1e-5`。batch分组引起微差不放宽容差；失败终止并保留已有证据。无需CSV字节相同。

成功后上传 `eeg_fulltrain_audit.zip`，失败上传 `eeg_fulltrain_partial_audit.zip` 和日志。包中含完整预测、分组、report、来源hash和audit_index，也保存新脚本自体、交付manifest与固定v1helper/source SHA。交付manifest校验本包四个源码/文档文件；新源码前后hash一致。三个检查点每完成一个打印短进度；run.sh最终只打印短摘要。

标准库测试（8项，不要求torch）：

```bash
cd /root/fatigue-eeg-fulltrain-v1-code
/root/miniconda3/bin/python -B -m unittest -v test_eeg_fulltrain.py
```

本地测试只验证分组/NA/匹配/路径保护与静态零训练约束，不代表服务器GPU评估通过。实际GPU通过必须等待新服务器report及ZIP审计。
