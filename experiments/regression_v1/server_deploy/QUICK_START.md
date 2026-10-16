# 🚀 服务器训练速查卡

## 📋 3步快速开始

### 1️⃣ 上传文件（本地终端）
```bash
cd "C:/Users/dingjianbo/Documents/ChatGPT/疲劳检测/work/regression_pilot/server_deploy"
scp *.py *.sh root@YOUR_SERVER:/root/regression-pilot-code/
```

### 2️⃣ SSH运行（服务器）
```bash
ssh root@YOUR_SERVER
cd /root/regression-pilot-code
chmod +x deploy_and_run.sh
bash deploy_and_run.sh
```

### 3️⃣ 等待并下载（1-2小时后，本地）
```bash
scp -r root@YOUR_SERVER:/root/regression-pilot-v1 ./results/
cat ./results/final_metrics.json
```

---

## ✅ 成功标准

| 指标 | 目标 | 当前基线 |
|------|------|---------|
| MAE | **<0.15** | N/A |
| 二分类准确率 | **>55%** | 49.70% |
| 改进 | **+5pp以上** | - |

---

## 📞 遇到问题？

**告诉我：**
1. 具体错误信息
2. 你执行到了哪一步
3. `ls -la /root/` 的输出

**常见问题：**
- 找不到模块 → 检查 `/root/Fatigue-check-Mamba-v2`
- 找不到数据 → 检查 `/root/autodl-tmp/...`
- 显存不足 → 改 `--batch-size 32`

---

## 📁 需要上传的文件（共3个）

✅ `regression_models.py` - 模型定义（7KB）  
✅ `regression_pilot.py` - 训练脚本（11KB）  
✅ `deploy_and_run.sh` - 部署脚本（3KB）

**位置：**
`C:/Users/dingjianbo/Documents/ChatGPT/疲劳检测/work/regression_pilot/server_deploy/`

---

## ⏱️ 预计时间

- 上传：1分钟
- 环境检查：1分钟
- **训练：1-2小时** ⏳
- 下载：1分钟

---

## 🎯 训练完成后

**立即通知我结果！发送：**
```bash
cat /root/regression-pilot-v1/final_metrics.json
```

**我会帮你：**
- 判断成功/失败
- 分析下一步
- 决定是否全面训练

---

**现在开始上传第一个文件吧！** 🚀

**完整指南：** 看 `SERVER_GUIDE.md`
