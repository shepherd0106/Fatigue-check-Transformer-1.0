"""
回归训练的最小修改指南

这个文件不是完整的训练脚本，而是展示如何修改现有的modality_experiment.py来支持回归。

核心修改点：
1. 模型：改用regression_models.py中的RegressionModalityModel
2. 损失：CrossEntropyLoss → MSELoss
3. 评估：添加回归指标 + 转换为分类评估
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

# ============================================================================
# 修改点1：模型初始化
# ============================================================================

# 原来（分类版本）：
# from modality_models import ModalityModel
# model = ModalityModel("C_E", seed=17, fold=2)

# 修改为（回归版本）：
# from regression_models import RegressionModalityModel
# model = RegressionModalityModel("C_E_regression", seed=17, fold=2)


# ============================================================================
# 修改点2：损失函数
# ============================================================================

# 原来（分类版本）：
# criterion = nn.CrossEntropyLoss()
# loss = criterion(logits, labels)  # logits: [batch, 2], labels: [batch] (0/1)

# 修改为（回归版本）：
# criterion = nn.MSELoss()
# loss = criterion(pred_F, target_F)  # pred_F: [batch], target_F: [batch] (0-1连续)


# ============================================================================
# 修改点3：训练循环
# ============================================================================

def train_one_epoch_regression(
    model, dataloader, optimizer, criterion, device
) -> Dict:
    """回归训练的一个epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        # 假设dataloader返回 (eeg, fnirs, label_continuous_F)
        eeg = batch['eeg'].to(device) if 'eeg' in batch else None
        fnirs = batch['fnirs'].to(device) if 'fnirs' in batch else None
        target_F = batch['label_mean_f'].to(device)  # 连续F值

        # 前向传播
        pred_F = model(eeg=eeg, fnirs=fnirs)  # [batch]

        # 计算损失
        loss = criterion(pred_F, target_F)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 记录
        total_loss += loss.item()
        all_predictions.extend(pred_F.detach().cpu().numpy())
        all_targets.extend(target_F.cpu().numpy())

    # 计算epoch指标
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    metrics = {
        'loss': total_loss / len(dataloader),
        'mae': np.mean(np.abs(predictions - targets)),
        'rmse': np.sqrt(np.mean((predictions - targets)**2))
    }

    return metrics


# ============================================================================
# 修改点4：评估函数（核心！）
# ============================================================================

def evaluate_regression_as_classification(
    predictions: np.ndarray,  # [N] 预测的连续F
    targets: np.ndarray,      # [N] 真实的连续F
    original_binary: np.ndarray,  # [N] 原始二分类标签
    original_ternary: np.ndarray, # [N] 原始三分类标签
    binary_threshold: float = 0.4,
    ternary_thresholds: Tuple[float, float] = (0.3, 0.6)
) -> Dict:
    """
    核心评估函数：从连续F预测转换为分类结果

    Returns:
        包含回归指标和分类指标的字典
    """
    metrics = {}

    # ===== 1. 回归指标 =====
    metrics['mae'] = np.mean(np.abs(predictions - targets))
    metrics['rmse'] = np.sqrt(np.mean((predictions - targets)**2))

    # 相关系数（需要scipy）
    try:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(predictions, targets)
        spearman_r, spearman_p = spearmanr(predictions, targets)
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p
    except:
        pass

    # ===== 2. 二分类指标 =====
    pred_binary = (predictions >= binary_threshold).astype(int)
    metrics['binary_accuracy'] = (pred_binary == original_binary).mean()

    # 按类别召回
    for class_id in [0, 1]:
        mask = (original_binary == class_id)
        if mask.sum() > 0:
            recall = (pred_binary[mask] == class_id).mean()
            metrics[f'binary_class{class_id}_recall'] = recall

    # ===== 3. 三分类指标 =====
    pred_ternary = np.zeros_like(predictions, dtype=int)
    pred_ternary[predictions >= ternary_thresholds[0]] = 1
    pred_ternary[predictions >= ternary_thresholds[1]] = 2

    metrics['ternary_accuracy'] = (pred_ternary == original_ternary).mean()

    # 按类别召回
    for class_id in [0, 1, 2]:
        mask = (original_ternary == class_id)
        if mask.sum() > 0:
            recall = (pred_ternary[mask] == class_id).mean()
            metrics[f'ternary_class{class_id}_recall'] = recall

    return metrics


# ============================================================================
# 修改点5：验证循环
# ============================================================================

def validate_regression(
    model, dataloader, criterion, device,
    binary_threshold=0.4, ternary_thresholds=(0.3, 0.6)
) -> Dict:
    """回归模型的验证"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_binary_labels = []
    all_ternary_labels = []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device) if 'eeg' in batch else None
            fnirs = batch['fnirs'].to(device) if 'fnirs' in batch else None
            target_F = batch['label_mean_f'].to(device)
            binary_label = batch['label_2class'].cpu().numpy()
            ternary_label = batch['label_3class'].cpu().numpy()

            # 前向传播
            pred_F = model(eeg=eeg, fnirs=fnirs)
            loss = criterion(pred_F, target_F)

            # 记录
            total_loss += loss.item()
            all_predictions.extend(pred_F.cpu().numpy())
            all_targets.extend(target_F.cpu().numpy())
            all_binary_labels.extend(binary_label)
            all_ternary_labels.extend(ternary_label)

    # 转换为numpy数组
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    binary_labels = np.array(all_binary_labels)
    ternary_labels = np.array(all_ternary_labels)

    # 计算所有指标
    metrics = evaluate_regression_as_classification(
        predictions, targets, binary_labels, ternary_labels,
        binary_threshold, ternary_thresholds
    )
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


# ============================================================================
# 修改点6：主训练循环
# ============================================================================

def train_regression_model(
    model, train_loader, val_loader, optimizer,
    num_epochs=100, patience=10, device='cuda'
):
    """完整的回归训练流程"""
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # 训练
        train_metrics = train_one_epoch_regression(
            model, train_loader, optimizer, criterion, device
        )

        # 验证
        val_metrics = validate_regression(
            model, val_loader, criterion, device
        )

        # 打印（每10个epoch）
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train: MAE={train_metrics['mae']:.4f}")
            print(f"  Val:   MAE={val_metrics['mae']:.4f}, "
                  f"Binary Acc={val_metrics['binary_accuracy']*100:.2f}%, "
                  f"Ternary Acc={val_metrics['ternary_accuracy']*100:.2f}%")

        # 早停
        if val_metrics['mae'] < best_val_mae - 1e-4:
            best_val_mae = val_metrics['mae']
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_regression_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    return best_val_mae


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("回归训练最小修改指南")
    print("="*70)
    print("\n这不是可运行的脚本，而是展示需要修改的关键代码段。")
    print("\n实际实现时，需要：")
    print("1. 复制 modality_v1/modality_experiment.py 为 regression_experiment.py")
    print("2. 按上述5个修改点逐一修改")
    print("3. 确保dataloader返回 label_mean_f（连续F）字段")
    print("4. 在evaluate函数中同时计算回归和分类指标")
    print("\n预计修改时间：30-60分钟")
    print("="*70)
