# 服务器训练完整操作指南

## 📦 文件清单

**本地准备（已生成）：**
```
C:/Users/dingjianbo/Documents/ChatGPT/疲劳检测/work/regression_pilot/server_deploy/
├── regression_models.py       # 回归模型定义
├── regression_pilot.py        # 独立训练脚本
├── deploy_and_run.sh          # 服务器部署脚本
└── SERVER_GUIDE.md            # 本文件
```

---

## 🚀 **快速开始（3步完成）**

### 第1步：上传文件到服务器（本地执行）

打开本地终端，执行：

```bash
# 切换到文件目录
cd "C:/Users/dingjianbo/Documents/ChatGPT/疲劳检测/work/regression_pilot/server_deploy"

# 上传到服务器
scp regression_models.py root@<your-server-ip>:/root/regression-pilot-code/
scp regression_pilot.py root@<your-server-ip>:/root/regression-pilot-code/
scp deploy_and_run.sh root@<your-server-ip>:/root/regression-pilot-code/

# 示例（假设服务器IP是123.45.67.89）：
# scp regression_models.py root@123.45.67.89:/root/regression-pilot-code/
# scp regression_pilot.py root@123.45.67.89:/root/regression-pilot-code/
# scp deploy_and_run.sh root@123.45.67.89:/root/regression-pilot-code/
```

**或者使用你熟悉的工具（WinSCP、FileZilla等）上传这3个文件。**

---

### 第2步：SSH连接到服务器并运行

```bash
# SSH连接
ssh root@<your-server-ip>

# 切换到代码目录
cd /root/regression-pilot-code

# 赋予脚本执行权限
chmod +x deploy_and_run.sh

# 运行部署脚本（会自动检查环境、数据，然后询问是否开始训练）
bash deploy_and_run.sh
```

---

### 第3步：等待结果并下载

训练完成后（1-2小时），下载结果：

```bash
# 在本地执行
scp -r root@<your-server-ip>:/root/regression-pilot-v1 ./results/

# 查看关键指标
cat ./results/final_metrics.json
```

---

## 📊 **预期输出示例**

### **训练过程输出：**
```
========================================================================
Starting training: C_E_regression, fold=2
========================================================================

Epoch 1 (step 64/2304, 12.3s)
  Train: loss=0.0523, MAE=0.1842
  Val:   MAE=0.1756, Binary=52.34%, Ternary=45.67%
  ✓ New best MAE: 0.1756

Epoch 5 (step 320/2304, 11.8s)
  Train: loss=0.0198, MAE=0.1123
  Val:   MAE=0.1234, Binary=58.92%, Ternary=51.23%
  ✓ New best MAE: 0.1234

...

Early stopping at epoch 23 (step 1472)

========================================================================
Final validation:
========================================================================

回归指标:
  MAE:  0.1187
  RMSE: 0.1542
  Pearson  r: 0.7823 (p=1.23e-234)
  Spearman r: 0.7654 (p=2.34e-221)

二分类指标 (阈值=0.4):
  准确率: 59.23%
  非疲劳召回: 62.15%
  疲劳召回:   56.87%

三分类指标 (阈值=0.3/0.6):
  准确率: 52.34%
  警觉召回: 58.92%
  过渡召回: 48.76%
  疲劳召回: 54.23%

与基线对比 (C_E分类: 49.70%):
  改进: +9.53pp

✅ 成功！达到目标标准

========================================================================
Training completed!
Results saved to: /root/regression-pilot-v1
========================================================================
```

---

## 🎯 **结果判断标准**

| 指标 | 优秀 | 成功 | 可行 | 失败 |
|------|------|------|------|------|
| **MAE** | <0.12 | 0.12-0.15 | 0.15-0.20 | >0.20 |
| **二分类准确率** | >60% | 55-60% | 50-55% | <50% |
| **判断** | 立即全面训练 | 全面训练+优化 | 需要分析 | 诊断or退回 |

**关键对比：**
- 当前C_E分类：49.70%
- 目标改进：至少+5pp（即>54.70%）

---

## 🐛 **常见问题与解决**

### **问题1：`ModuleNotFoundError: No module named 'fatigue_training'`**

**原因：** 服务器上没有主代码库

**解决：**
```bash
# 检查主代码库是否存在
ls -la /root/Fatigue-check-Mamba-v2

# 如果不存在，需要先部署主代码库
cd /root
git clone <your-repo-url> Fatigue-check-Mamba-v2
```

---

### **问题2：`FileNotFoundError: manifest.json not found`**

**原因：** 数据路径不对

**解决：**
```bash
# 检查数据是否存在
ls -la /root/autodl-tmp/fatigue_mamba_data_v2/fatigue_time_axis_v2_preview/

# 如果路径不同，修改regression_pilot.py中的manifest_path
```

---

### **问题3：CUDA out of memory**

**原因：** 显存不足

**解决：**
```bash
# 方法1：减小batch size
python3 regression_pilot.py --batch-size 32 --execute

# 方法2：清理显存
nvidia-smi
kill <占用GPU的进程ID>
```

---

### **问题4：训练很慢，一直卡在某个epoch**

**原因：** 可能是数据加载问题

**解决：**
```bash
# 检查数据完整性
cd /root/autodl-tmp/fatigue_mamba_data_v2/fatigue_time_axis_v2_preview
ls -lh

# 检查是否有损坏的文件
python3 -c "
from fatigue_training.data import WindowDataset
dataset = WindowDataset('manifest.json')
print(f'Total windows: {len(dataset)}')
"
```

---

## 📁 **服务器文件结构（训练后）**

```
/root/
├── regression-pilot-code/          # 代码目录
│   ├── regression_models.py
│   ├── regression_pilot.py
│   └── deploy_and_run.sh
│
├── regression-pilot-v1/            # 结果目录
│   ├── best_model.pt               # 最佳模型权重
│   ├── best_metrics.json           # 最佳指标
│   ├── final_model.pt              # 最终模型权重
│   ├── final_metrics.json          # 最终指标
│   └── training.log                # 完整训练日志
│
├── Fatigue-check-Mamba-v2/         # 主代码库（已有）
│   └── fatigue_training/
│       ├── development.py
│       ├── data.py
│       └── normalization.py
│
└── autodl-tmp/                     # 数据目录（已有）
    └── fatigue_mamba_data_v2/
        └── fatigue_time_axis_v2_preview/
            ├── manifest.json
            ├── windows.csv
            └── runs/
```

---

## 🔄 **训练完成后的下一步**

### **如果成功（MAE<0.15, 准确率>55%）：**

1. **立即通知我结果**，我会帮你：
   - 分析为什么回归比分类好
   - 准备全面训练计划（四折×三模型）
   - 设计PVT验证实验

2. **保留checkpoint**：
   ```bash
   # 不要删除 /root/regression-pilot-v1/
   # 后续可能需要加载权重做分析
   ```

### **如果失败（MAE>0.20或准确率<50%）：**

1. **下载完整日志给我分析**：
   ```bash
   scp root@server:/root/regression-pilot-v1/training.log ./
   ```

2. **我会帮你诊断**：
   - 检查训练曲线是否正常
   - 分析预测分布
   - 决定是调整超参数还是退回方案A

---

## ⏱️ **时间估算**

| 阶段 | 时间 | 说明 |
|------|------|------|
| 文件上传 | 1分钟 | 3个小文件 |
| 环境检查 | 1分钟 | 自动脚本 |
| Dry run测试 | 2分钟 | 不实际训练 |
| **实际训练** | **1-2小时** | 取决于GPU（RTX5090约1小时） |
| 结果下载 | 1分钟 | JSON+模型文件 |
| **总计** | **约1-2小时** | |

---

## ✅ **执行检查清单**

**开始前：**
- [ ] 3个文件已上传到服务器
- [ ] SSH能正常连接服务器
- [ ] 服务器上有 `/root/Fatigue-check-Mamba-v2/`
- [ ] 服务器上有数据 `/root/autodl-tmp/fatigue_mamba_data_v2/`

**训练中：**
- [ ] 看到 "Starting training" 消息
- [ ] 每5个epoch打印一次指标
- [ ] MAE逐渐下降
- [ ] 没有CUDA out of memory错误

**训练后：**
- [ ] 看到 "Training completed!" 消息
- [ ] `/root/regression-pilot-v1/` 目录存在
- [ ] `final_metrics.json` 文件存在
- [ ] 已下载结果到本地

---

## 💬 **需要帮助？**

**训练前遇到问题：**
- 告诉我具体的错误信息
- 发送 `ls -la /root/` 的输出
- 发送 `python3 regression_pilot.py` 的输出

**训练后需要分析：**
- 发送 `final_metrics.json` 的内容
- 发送 `training.log` 的最后100行
- 告诉我你的判断（成功/失败）

**我会在这里等你的反馈！** 🚀

---

## 🎯 **最简化的3条命令（如果你很熟悉服务器）**

```bash
# 1. 上传（本地）
cd "C:/Users/dingjianbo/Documents/ChatGPT/疲劳检测/work/regression_pilot/server_deploy"
scp *.py *.sh root@<server>:/root/regression-pilot-code/

# 2. 训练（服务器）
ssh root@<server>
cd /root/regression-pilot-code && chmod +x deploy_and_run.sh && bash deploy_and_run.sh

# 3. 下载（本地）
scp -r root@<server>:/root/regression-pilot-v1 ./results/
```

---

**准备好了吗？开始上传文件吧！** 🚀
