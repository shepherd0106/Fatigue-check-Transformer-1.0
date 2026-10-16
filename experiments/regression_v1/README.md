# 方案B：回归训练 + 二分类/三分类输出

## 目标
用连续F回归训练模型，测试时转换为二分类和三分类结果，验证是否比直接分类训练更有效。

## 核心改动

### 1. 模型输出层
```python
# 原来：输出2个logits（二分类）
output = Linear(hidden_dim, 2)

# 现在：输出1个连续值（0-1之间的F）
output = Linear(hidden_dim, 1) + Sigmoid
```

### 2. 损失函数
```python
# 原来：CrossEntropyLoss
loss = CELoss(logits, binary_labels)

# 现在：MSE Loss
loss = MSELoss(predicted_F, target_continuous_F)
```

### 3. 评估指标
**训练时监控：**
- 回归指标：MAE, RMSE, Pearson相关系数
- 二分类准确率（阈值0.4）
- 三分类准确率（阈值0.3/0.6）

**最终报告：**
- 回归：MAE, RMSE, Spearman/Pearson相关
- 二分类：准确率、非疲劳召回、疲劳召回
- 三分类：准确率、三类召回率

## Pilot实验设置

**模型：** C_E (EEG-only CNN, 1802参数)  
**配置：** fold2, seed17, lr=0.001, dropout=0.3  
**训练步数：** 2304 (与D00对照保持一致)  
**标签：** 连续F值（从当前windows CSV读取label_mean_f列）

**对比基线：**
- 当前C_E二分类结果：主指标49.70%
- 当前C_E fold2验证集：3856个窗口

## 成功标准

**最低目标（证明可行）：**
- MAE < 0.20
- 二分类准确率 > 50%

**理想目标（证明优势）：**
- MAE < 0.15
- 二分类准确率 > 55%
- 与当前标签的相关系数 > 0.7

**优秀目标（显著改善）：**
- MAE < 0.12
- 二分类准确率 > 60%
- 三分类准确率 > 50%

## 下一步决策

### 如果Pilot成功（MAE<0.15, 准确率>55%）
1. 全面训练：四折 × 三模型(C_E/C_F/C_EF) × seed17 = 12 fit
2. 加入seed29/43
3. 测试融合模块（Q/A）
4. PVT验证分析

### 如果Pilot失败（MAE>0.20或准确率<50%）
1. 检查预测分布（是否过于保守/激进）
2. 尝试不同损失函数（Huber Loss, Focal Loss for Regression）
3. 如果仍失败，退回到方案A（保守优化当前分类模型）

## 时间表

- **今天下午**：实现代码，准备配置
- **今晚/明早**：运行pilot实验（约1-2小时GPU时间）
- **明天下午**：分析结果，决定下一步
- **本周内**：完成至少一轮完整实验（成功或失败都有明确结论）

## 文件结构

```
work/regression_pilot/
├── README.md                          # 本文件
├── pilot_config.json                  # 实验配置
├── regression_wrapper.py              # 回归+分类评估包装器
└── modified_training_code/            # 修改后的训练代码（待生成）
    ├── regression_models.py           # 回归版本的模型
    ├── regression_training.py         # 回归训练循环
    └── run_pilot.sh                   # 运行脚本
```
