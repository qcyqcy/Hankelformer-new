"""
极端事件评估指标计算脚本 - 无需原始尺度版

本脚本计算的指标基于相对排名和比例，不受数据归一化影响。

使用方法:
    python compute_extreme_metrics_scale_independent.py --pred_path <pred.npy路径> --true_path <true.npy路径>

示例:
    python compute_extreme_metrics_scale_independent.py --pred_path "results/.../pred.npy" --true_path "results/.../true.npy"

================================================================================
                              包含的指标
================================================================================

1. SEDI (Symmetric Extreme Dependency Index)
   - 对称极值依赖指数
   - 评估模型预测极端高温/低温的能力
   - 范围 [0, 1]，越接近1越好
   - 基于相对排名计算，不受归一化影响

2. exceedance_skill (Threshold Exceedance Skill)
   - 阈值超标技能得分
   - 评估模型预测超过某一阈值的极端事件的能力
   - 1为完美，0表示和 climatology 相当，>1 表示预测偏多，<1 表示预测偏少
   - 基于比例计算，不受归一化影响

================================================================================
"""

import argparse
import numpy as np


def SEDI(pred, true, percentile=95):
    """
    对称极值依赖指数 (Symmetric Extreme Dependency Index)
    用于评估模型预测极端高温/低温的能力

    参数:
        pred: 预测值
        true: 真实值
        percentile: 用于定义极端值的百分位阈值 (默认95%)

    返回:
        sedi: SEDI 值，范围 [0, 1]，越接近1越好
    """
    threshold = np.percentile(true, percentile)

    extreme_true = true[true >= threshold]
    extreme_pred = pred[true >= threshold]

    if len(extreme_true) == 0:
        return np.nan

    pred_exceed = (pred >= threshold).sum() / len(pred)
    true_exceed = (true >= threshold).sum() / len(true)

    if pred_exceed == 0 or true_exceed == 0:
        return 0.0

    sedi = 2 * (pred_exceed - true_exceed) / (pred_exceed + true_exceed)
    sedi = 1 - abs(sedi)

    return sedi


def threshold_exceedance_skill(pred, true, threshold_percentile=90):
    """
    阈值超标技能得分 (Threshold Exceedance Skill)

    用于评估模型预测超过某一阈值的极端事件的能力

    参数:
        pred: 预测值
        true: 真实值
        threshold_percentile: 定义阈值的百分位 (基于真实值分布)

    返回:
        skill: 技能得分，1为完美，0表示和 climatology 相当，
              >1 表示预测偏多，<1 表示预测偏少
    """
    threshold = np.percentile(true, threshold_percentile)

    pred_exceed = (pred >= threshold).sum() / len(pred)
    true_exceed = (true >= threshold).sum() / len(true)

    if true_exceed == 0:
        skill = 1.0 if pred_exceed == 0 else 0.0
    else:
        skill = pred_exceed / true_exceed

    return skill


def compute_scale_independent_metrics(pred, true, percentile=95, threshold_percentile=90):
    """
    计算不依赖原始尺度的极端事件指标

    参数:
        pred: 预测值, shape [num_samples, pred_len, num_vars] 或展平的一维数组
        true: 真实值, shape 同 pred
        percentile: SEDI 使用
        threshold_percentile: 阈值超标技能得分使用

    返回:
        dict: 包含所有指标
    """
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    return {
        'SEDI': SEDI(pred_flat, true_flat, percentile),
        'exceedance_skill': threshold_exceedance_skill(pred_flat, true_flat, threshold_percentile),
    }


def print_metrics(metrics, title="极端事件评估结果 (无需原始尺度)"):
    """打印指标结果"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"  SEDI (95%):             {metrics['SEDI']:.6f}")
    print(f"  exceedance_skill (90%): {metrics['exceedance_skill']:.6f}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='计算极端事件评估指标 (无需原始尺度)')
    parser.add_argument('--pred_path', type=str, required=True, help='pred.npy 文件路径')
    parser.add_argument('--true_path', type=str, required=True, help='true.npy 文件路径')
    parser.add_argument('--percentile', type=int, default=95, help='SEDI 百分位阈值')
    parser.add_argument('--threshold_percentile', type=int, default=90, help='阈值超标技能得分的百分位阈值')

    args = parser.parse_args()

    print(f"加载预测: {args.pred_path}")
    print(f"加载真实: {args.true_path}")
    pred = np.load(args.pred_path)
    true = np.load(args.true_path)

    print(f"形状: pred={pred.shape}, true={true.shape}")

    metrics = compute_scale_independent_metrics(
        pred, true,
        percentile=args.percentile,
        threshold_percentile=args.threshold_percentile
    )
    print_metrics(metrics)


if __name__ == '__main__':
    main()