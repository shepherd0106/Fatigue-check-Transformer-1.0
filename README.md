# EEG–fNIRS fatigue classification

本仓库包含早期CNN＋Transformer框架，以及已完成开发实验的CNN＋Mamba-1候选、训练诊断和单模态工具。数据、权重与服务器结果单独保存。当前结果均有各自的开发/训练内范围，数据标签、时间对应与光学科学来源仍有未核实项。

## 代码入口

| 目录 | 用途 | 状态 |
|---|---|---|
| `fatigue_training/development*.py` | `cnn_mamba_fusion_development_v2`：C/M0/T/Q/A/F/K及S | 已完成116研究fit；BN1 bias统一固定0 |
| `fatigue_training/models.py`、`configs/protocol.json` | 旧Transformer协议与可复用CNN/分组定义 | 历史协议，不能把945fit计划当作已完成实验 |
| `diagnostics_v1/` | tiny和Constant/Cosine固定长训诊断 | 独立6run/16384步，已有服务器结果 |
| `modality_v1/` | C_E/C_F单模态CNN及匹配EF参照 | 独立12run、含8研究fit，已有结果 |
| `tools/EEG只读诊断_20261016/code/` | 既有权重和probe的只读诊断 | 0新增fit，已有结果 |
| `tools/EEG完整训练评估_20261016/code/` | 全11687训练窗只读前向 | 0新增fit，已有结果 |
| `tools/EEG拟合对照_20261016/code/` | 新固定576窗Dropout 0.3/0对照 | 2fit/4608步已完成，不自动重跑 |
| `experiments/regression_v1/` | 后续连续F回归草稿 | 接口尚不兼容，见STATUS.md；未作为可运行入口验收 |
| `tests/` | 原协议、development、BN修订和时间轴回归测试 | 缺Torch的项目会明确skip |

## 模型与数据

10秒窗、5秒步长；EEG为4×2500@250Hz，fNIRS为24×250@25Hz（8组HbO/HbR/HbT）。当前新开发在24人中按18/6作四内层验证，当前轮隔离六人[2,9,12,20,29,33]；这六人在更早Transformer阶段已经被观察。

完整候选F：两路独立CNN→各两层Mamba-1→全窗＋四个有序区段读出Q→fNIRS条件仿射调制EEG的A→拼接分类。可训练参数54794。Mamba架构为1，软件包版本为2.2.6.post3，两者含义不同。

开发配置由`fatigue_training.development.configuration()`生成，复用旧CNN/分组后覆写训练设置；不存在单独的development_v2.json。新协议训练标准化在`development_features.fit_normalization`（训练人覆盖去重/人等权、零方差scale=1）；旧`normalization.py`采用方差floor，两者不混用。

## 安装和只读查看

Python>=3.10，NumPy>=1.26；模型需要合适的PyTorch/CUDA。服务器已验证过Torch2.8.0+cu128与官方Mamba源码commit `10b5d6358f27966f6a40e4bf0baa17a460688128`。先核对目标环境，不因整理仓库自动重装已有依赖。

```bash
python -m fatigue_training
python -m unittest discover -s tests -v
```

训练、预处理和部署需要相应数据及明确协议/预算。已有输出的防覆盖、预算及writer锁应保留，Git上传本身不启动任何实验。完整CLI、数据合同和历史设计说明保留于[旧README](docs/archive/README_before_20261016.md)，其中“GPU待验证/尚未训练”等段落是当时状态；当前已完成范围见[项目状态](docs/PROJECT_STATE.md)。

## 版本与保存

`tools/`保留已交付脚本的原字节与manifest，避免改变历史证据。`.gitattributes`禁止自动换行转换，以保持这些文件及冻结源码的SHA绑定。原始信号、检查点、环境、结果包及本机清理/应用迁移脚本不纳入仓库。

新草稿与已完成实验分别记录；模型与指标的改动应另立协议，不能用新代码覆写旧结果的验收状态。
