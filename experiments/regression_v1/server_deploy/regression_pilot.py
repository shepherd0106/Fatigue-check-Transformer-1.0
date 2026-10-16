#!/usr/bin/env python3
"""
回归训练Pilot实验 - 服务器独立运行版本

使用方式：
    python regression_pilot.py --execute

说明：
- 独立脚本，不依赖复杂的modality框架
- 直接使用development.py的数据加载和配置
- 专注于快速验证回归方案
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 路径适配：优先服务器路径，本地测试用D盘
SERVER_ROOT = Path("/root/Fatigue-check-Mamba-v2")
LOCAL_ROOT = Path("D:/Fatigue check")

if SERVER_ROOT.exists():
    code_root = SERVER_ROOT
elif LOCAL_ROOT.exists():
    code_root = LOCAL_ROOT
else:
    raise RuntimeError("Cannot find code root (neither server nor local)")

sys.path.insert(0, str(code_root))
print(f"Using code root: {code_root}")

from fatigue_training.development import configuration, subjects
from fatigue_training.data import WindowDataset
from fatigue_training.normalization import fit_normalization, normalize_batch
from regression_models import RegressionModalityModel


class RegressionTrainer:
    """回归训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载数据
        self.load_data()

        # 创建模型
        self.model = RegressionModalityModel(args.variant, seed=args.seed, fold=args.fold)
        self.model.to(self.device)
        print(f"Model: {args.variant}, params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # 损失和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 训练状态
        self.best_mae = float('inf')
        self.patience_counter = 0
        self.global_step = 0

    def load_data(self):
        """加载数据集"""
        config = configuration()
        train_subs, val_subs = subjects(self.args.fold)

        # 加载数据集
        manifest_path = Path("/root/autodl-tmp/fatigue_mamba_data_v2/fatigue_time_axis_v2_preview/manifest.json")
        dataset = WindowDataset(str(manifest_path))
        dataset.check_cohort(config)

        # 分割训练集和验证集
        train_indices = dataset.indices(train_subs)
        val_indices = dataset.indices(val_subs)

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)

        print(f"Train: {len(train_indices)} windows from {len(train_subs)} subjects")
        print(f"Val:   {len(val_indices)} windows from {len(val_subs)} subjects")

        # 拟合标准化参数
        print("Fitting normalization...")
        self.normalization_stats = fit_normalization(dataset, train_subs)

        # 创建DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        losses = []
        maes = []

        for batch in self.train_loader:
            # 检查步数上限
            if self.global_step >= self.args.max_steps:
                break

            # 数据准备
            eeg = batch['eeg'].to(self.device) if 'eeg' in batch else None
            fnirs = batch['fnirs'].to(self.device) if 'fnirs' in batch else None
            target_F = batch['label_mean_f'].to(self.device)

            # 标准化
            if eeg is not None:
                eeg = normalize_batch(eeg, self.normalization_stats['eeg'])
            if fnirs is not None:
                fnirs = normalize_batch(fnirs, self.normalization_stats['fnirs'])

            # 前向传播
            if 'eeg' in self.args.variant.lower():
                pred_F = self.model(eeg=eeg)
            else:
                pred_F = self.model(fnirs=fnirs)

            loss = self.criterion(pred_F, target_F)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 记录
            losses.append(loss.item())
            maes.append(torch.abs(pred_F - target_F).mean().item())
            self.global_step += 1

        return {
            'loss': np.mean(losses),
            'mae': np.mean(maes)
        }

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_binary_labels = []
        all_ternary_labels = []
        losses = []

        for batch in self.val_loader:
            eeg = batch['eeg'].to(self.device) if 'eeg' in batch else None
            fnirs = batch['fnirs'].to(self.device) if 'fnirs' in batch else None
            target_F = batch['label_mean_f'].to(self.device)

            # 标准化
            if eeg is not None:
                eeg = normalize_batch(eeg, self.normalization_stats['eeg'])
            if fnirs is not None:
                fnirs = normalize_batch(fnirs, self.normalization_stats['fnirs'])

            # 前向传播
            if 'eeg' in self.args.variant.lower():
                pred_F = self.model(eeg=eeg)
            else:
                pred_F = self.model(fnirs=fnirs)

            loss = self.criterion(pred_F, target_F)

            # 收集
            losses.append(loss.item())
            all_preds.extend(pred_F.cpu().numpy())
            all_targets.extend(target_F.cpu().numpy())
            all_binary_labels.extend(batch['label_2class'].numpy())
            all_ternary_labels.extend(batch['label_3class'].numpy())

        # 转换为numpy
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        binary = np.array(all_binary_labels)
        ternary = np.array(all_ternary_labels)

        # 计算指标
        metrics = self.compute_metrics(preds, targets, binary, ternary)
        metrics['loss'] = np.mean(losses)

        return metrics

    def compute_metrics(self, preds, targets, binary, ternary):
        """计算回归和分类指标"""
        metrics = {}

        # 回归指标
        metrics['mae'] = np.mean(np.abs(preds - targets))
        metrics['rmse'] = np.sqrt(np.mean((preds - targets)**2))

        # 相关系数
        try:
            from scipy.stats import pearsonr, spearmanr
            metrics['pearson_r'], metrics['pearson_p'] = pearsonr(preds, targets)
            metrics['spearman_r'], metrics['spearman_p'] = spearmanr(preds, targets)
        except:
            pass

        # 二分类（阈值0.4）
        pred_binary = (preds >= 0.4).astype(int)
        metrics['binary_accuracy'] = (pred_binary == binary).mean()
        for c in [0, 1]:
            mask = (binary == c)
            if mask.sum() > 0:
                metrics[f'binary_class{c}_recall'] = (pred_binary[mask] == c).mean()

        # 三分类（阈值0.3/0.6）
        pred_ternary = np.zeros_like(preds, dtype=int)
        pred_ternary[preds >= 0.3] = 1
        pred_ternary[preds >= 0.6] = 2
        metrics['ternary_accuracy'] = (pred_ternary == ternary).mean()
        for c in [0, 1, 2]:
            mask = (ternary == c)
            if mask.sum() > 0:
                metrics[f'ternary_class{c}_recall'] = (pred_ternary[mask] == c).mean()

        return metrics

    def train(self):
        """完整训练流程"""
        print("\n" + "="*70)
        print(f"Starting training: {self.args.variant}, fold={self.args.fold}")
        print("="*70)

        epoch = 0
        while self.global_step < self.args.max_steps:
            epoch += 1
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            epoch_time = time.time() - start_time

            # 打印
            if epoch % 5 == 0 or epoch == 1:
                print(f"\nEpoch {epoch} (step {self.global_step}/{self.args.max_steps}, {epoch_time:.1f}s)")
                print(f"  Train: loss={train_metrics['loss']:.4f}, MAE={train_metrics['mae']:.4f}")
                print(f"  Val:   MAE={val_metrics['mae']:.4f}, "
                      f"Binary={val_metrics['binary_accuracy']*100:.2f}%, "
                      f"Ternary={val_metrics['ternary_accuracy']*100:.2f}%")

            # 早停
            if val_metrics['mae'] < self.best_mae - 1e-4:
                self.best_mae = val_metrics['mae']
                self.patience_counter = 0
                # 保存最佳模型
                self.save_checkpoint(val_metrics, 'best')
                print(f"  ✓ New best MAE: {self.best_mae:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.args.patience:
                    print(f"\nEarly stopping at epoch {epoch} (step {self.global_step})")
                    break

        # 最终验证
        print("\n" + "="*70)
        print("Final validation:")
        print("="*70)
        final_metrics = self.validate()
        self.save_checkpoint(final_metrics, 'final')
        self.print_final_results(final_metrics)

        return final_metrics

    def save_checkpoint(self, metrics, name):
        """保存检查点"""
        output_dir = Path(self.args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step,
            'args': vars(self.args)
        }

        torch.save(checkpoint, output_dir / f'{name}_model.pt')

        # 保存metrics为JSON
        with open(output_dir / f'{name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    def print_final_results(self, metrics):
        """打印最终结果"""
        print(f"\n回归指标:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        if 'pearson_r' in metrics:
            print(f"  Pearson  r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
            print(f"  Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")

        print(f"\n二分类指标 (阈值=0.4):")
        print(f"  准确率: {metrics['binary_accuracy']*100:.2f}%")
        print(f"  非疲劳召回: {metrics.get('binary_class0_recall', 0)*100:.2f}%")
        print(f"  疲劳召回:   {metrics.get('binary_class1_recall', 0)*100:.2f}%")

        print(f"\n三分类指标 (阈值=0.3/0.6):")
        print(f"  准确率: {metrics['ternary_accuracy']*100:.2f}%")
        print(f"  警觉召回: {metrics.get('ternary_class0_recall', 0)*100:.2f}%")
        print(f"  过渡召回: {metrics.get('ternary_class1_recall', 0)*100:.2f}%")
        print(f"  疲劳召回: {metrics.get('ternary_class2_recall', 0)*100:.2f}%")

        print(f"\n与基线对比 (C_E分类: 49.70%):")
        improvement = (metrics['binary_accuracy'] - 0.497) * 100
        print(f"  改进: {improvement:+.2f}pp")

        # 判断成功或失败
        if metrics['mae'] < 0.15 and metrics['binary_accuracy'] > 0.55:
            print(f"\n✅ 成功！达到目标标准")
        elif metrics['mae'] < 0.20 and metrics['binary_accuracy'] > 0.50:
            print(f"\n⚠️  可行但未达最佳，需要分析和优化")
        else:
            print(f"\n❌ 未达标准，需要诊断问题或退回方案A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='C_E_regression', choices=['C_E_regression', 'C_F_regression'])
    parser.add_argument('--fold', type=int, default=2, choices=[2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-steps', type=int, default=2304)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output', default='/root/regression-pilot-v1')
    parser.add_argument('--execute', action='store_true', help='Actually run training')

    args = parser.parse_args()

    if not args.execute:
        print("Dry run mode. Use --execute to actually train.")
        print(f"Config: {args}")
        return

    # 训练
    trainer = RegressionTrainer(args)
    metrics = trainer.train()

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Results saved to: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()
