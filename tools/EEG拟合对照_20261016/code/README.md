# EEG 576窗Dropout拟合对照

只执行授权的两fit：D03三处StreamDropout p=0.3；D00 p=0。模块全名为`eeg.front.dropout1`、`eeg.front.dropout2`、顶层`dropout`。C_E/fold2/seed17，1802可训练参数，两臂与原未训练EF对应分支初始化一致，所有初始参数与buffers逐张量hash相同；只改变p，不改变网络、BN策略或参数量。

训练窗口是固定候选576（18人×每类16），覆盖12 block+4 rest；不同于原probe576。沿用原18人归一化，不重估统计。候选canonical SHA `347e4082a314646d76711688d8151a2a64cd8e420b21a3a3be1cf43d4d31e30c`，JSON字节SHA `ae186aa44ad1fecd2c325f7f39206f282b7fab52523c59e6a563db49fc389ab7`，逐条核训练index/source/人/类/run/窗序号及全阶段覆盖；不会按模型分数改选窗口。

每臂固定2304更新，batch64，无早停和选best端点，无额外优化预检。AdamW lr0.001 Constant，betas(0.9,0.999)、eps1e-8，原decay/no_decay组（decay=.001）、clip1，FP32、不使用AMP/TF32、原确定性设置。B1/B2/B64真实候选输入shape/finite检查只用eval，不耗优化步；原m.update保留梯度finite、冻结BN1bias与clip检查。

共享均匀有放回采样：`stream_seed(17,2,'eeg_fit576_dropout_v1')=3220531314883987093`；NumPy default_rng.integers(576,size=(2304,64))映射固定候选顺序。int64 NPY SHA锁定`504ba9ea56220fc5f986b9bf924c9af0ab0dce37cd666797a9dd3a313334eba7`。两臂逐步使用同一stream；总147456抽样、每窗曝光213–301、均值256，实际label draw为73588/73868。StreamDropout各处沿用原隔离seed，p0仍增加calls；两臂BN统计允许随训练分化。完整与实际执行前缀曝光按窗/人/run/段/类留档。

新预算独立2runs/4608优化步。原m.update在backward/检查之后，optimizer.step之前调用新Budget.reserve；更新正常返回后confirm。异常可能留下1个已预留但未确认步，额度不退还，报告上界与已确认量。原预算不变；失败/中断不重试、不续跑。训练CLI只接受唯一协议输出`/root/fatigue-eeg-fit576-v1`，不能更换output重建预算；输出拒绝任何已存在目录，writer.lock单实例。不可强杀后删除证据重跑。

部署本目录到`/root/fatigue-eeg-fit576-v1-code`，不要覆盖任何旧code/results目录。无需安装依赖。需要已有原模态code/results、已审readonly code/results及fulltrain code/results。helper导入前SHA固定，fulltrain audit_index固定为`79d240aef0ade6b3b9649a9fdc9cda1e9d1d1434c53b79a843f30a71f3b94ec3`；原source/results/baseline/旧ledger/全部完成工件、前序审计证据、训练EEG+coverage来源前后核验。新输出禁止与任何这些目录、数据、预检目录或新code重叠。

前台启动（两臂一次完成）：

```bash
bash /root/fatigue-eeg-fit576-v1-code/run.sh
```

脚本使用`/root/miniconda3/bin/python`，固定cuda:0，无CPU回退。输出`/root/fatigue-eeg-fit576-v1`，日志`/root/fatigue-eeg-fit576-v1.log`，已有日志也拒绝覆盖。日志每64步短进度；成功打印摘要，失败打印最后25行并保留退出码。另一终端可只读查看：

```bash
/root/miniconda3/bin/python -B /root/fatigue-eeg-fit576-v1-code/check_fit.py
```

每64步对新训练576输出逐窗logits和原按人primary/accuracy/CE/两类recall/预测疲劳率，记录最后64步train CE。原evaluate_fixed校验BN/buffers与RNG无变化。每臂保存initial.pt、last.pt（模型与optimizer状态）；只比较固定2304端点。端点重载复现训练576，另对11687完整训练窗做一次前向，输出新576/其余11111/原tiny128/原probe576、按人/16阶段/按run统计；完整前向抽取新576再固定rtol1e-4、atol1e-5复现，不放宽容差。

新576与其余11111按窗口编号互斥，但时间上可能重叠；原tiny和probe另为诊断参照且与新576交叉。所有范围均是已观察训练人，包含原归一化共享，不称validation、独立测试或formal验收；没有验证人、外层人或fNIRS信号推理。结果只回答该固定设置是否能拟合及关闭Dropout如何改变路径，不能单独定位BN或容量原因。

成功上传`eeg_fit576_audit.zip`，失败上传`eeg_fit576_partial_audit.zip`及日志。ZIP保留`runs/D03`/`runs/D00`层级，包含源码/manifest/binding/selection/ledger/curve/所有预测/分组/来源hash；不含pt/npy，服务器原件保留，其SHA在各臂status或binding中。输出初始化后的常规异常/SIGTERM/Ctrl-C尽力生成partial审计；kill -9、断电或磁盘写满可能阻止导出，应保留原目录与日志交还审查，不自动恢复。

标准库测试：

```bash
cd /root/fatigue-eeg-fit576-v1-code
/root/miniconda3/bin/python -B -m unittest -v test_eeg_fit576.py
```

测试覆盖预算持久化、失败不退款/重试、步数上限、无副作用的非法完成拒绝、sampling前缀、候选元数据匹配、路径保护与固定训练输出、层级ZIP及只读进度。若report已写但打包失败，check_fit优先报告收尾失败，不宣称成功上传；只有无failure、completed report与成功ZIP同时存在才提示上传。实际模型p-only测试需torch和已部署原源，无这些条件明确skip；本机标准库测试不等于服务器GPU或两fit完成。
