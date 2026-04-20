"""
极端事件评估指标计算脚本

使用方法:
    python compute_extreme_metrics.py --pred_path <pred.npy路径> --true_path <true.npy路径>

示例:
    python compute_extreme_metrics.py --pred_path "test_results/.../pred.npy" --true_path "test_results/.../true.npy"

注意:
    pred.npy 和 true.npy 必须是 inverse transform 后的原始尺度数据。
    运行测试时请加上 --inverse 参数，否则 peak_amplitude_error（峰值振幅误差）
    将无法得到有物理意义的数值（°C）。
"""

import argparse
import numpy as np


# ============ 极端事件评估指标 ============

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


def peak_amplitude_error(pred, true, mode='max'):
    """
    峰值振幅误差：预测峰值与实际峰值的直接对比

    参数:
        pred: 预测值
        true: 真实值
        mode: 'max'(最大峰值) 或 'min'(最小峰值) 或 'range'(峰值范围)

    返回:
        peak_err: 峰值振幅误差 (物理单位)
    """
    if mode == 'max':
        pred_peak = np.max(pred)
        true_peak = np.max(true)
    elif mode == 'min':
        pred_peak = np.min(pred)
        true_peak = np.min(true)
    else:
        pred_peak = np.max(pred) - np.min(pred)
        true_peak = np.max(true) - np.min(true)

    return pred_peak - true_peak


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


def extreme_MAE(pred, true, top_k=10):
    """
    极端峰值平均绝对误差 (以℃为单位)

    只针对极端高温/低温事件计算 MAE

    参数:
        pred: 预测值
        true: 真实值
        top_k: 计算误差的极端值数量（取top-k个最极端的值）

    返回:
        extreme_mae: 极端值的MAE
    """
    mean_true = np.mean(true)
    deviation = np.abs(true - mean_true)

    top_k = min(top_k, len(deviation))
    extreme_indices = np.argsort(deviation)[-top_k:]

    extreme_true = true[extreme_indices]
    extreme_pred = pred[extreme_indices]

    return np.mean(np.abs(extreme_true - extreme_pred))


def compute_extreme_metrics(pred, true, percentile=95, top_k=10, threshold_percentile=90):
    """
    计算极端事件指标

    参数:
        pred: 预测值, shape [num_samples, pred_len, num_vars]
        true: 真实值, shape 同 pred
        percentile: SEDI 使用
        top_k: 极端MAE使用
        threshold_percentile: 阈值超标技能得分使用

    返回:
        dict: 包含所有极端事件指标
    """
    return {
        'SEDI': SEDI(pred, true, percentile),
        'peak_max_error': peak_amplitude_error(pred, true, mode='max'),
        'peak_min_error': peak_amplitude_error(pred, true, mode='min'),
        'peak_range_error': peak_amplitude_error(pred, true, mode='range'),
        'exceedance_skill': threshold_exceedance_skill(pred, true, threshold_percentile),
        'extreme_MAE': extreme_MAE(pred, true, top_k),
    }


def print_metrics(metrics, title):
    """打印指标结果"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"  SEDI (95%):              {metrics['SEDI']:.6f}")
    print(f"  peak_max_error (℃):     {metrics['peak_max_error']:.6f}")
    print(f"  peak_min_error (℃):     {metrics['peak_min_error']:.6f}")
    print(f"  peak_range_error (℃):   {metrics['peak_range_error']:.6f}")
    print(f"  exceedance_skill (90%):  {metrics['exceedance_skill']:.6f}")
    print(f"  extreme_MAE (℃):       {metrics['extreme_MAE']:.6f}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='计算极端事件评估指标')
    parser.add_argument('--pred_path', type=str, required=True, help='pred.npy 文件路径')
    parser.add_argument('--true_path', type=str, required=True, help='true.npy 文件路径')
    parser.add_argument('--percentile', type=int, default=95, help='SEDI 百分位阈值')
    parser.add_argument('--top_k', type=int, default=10, help='极端MAE的top-k个数')
    parser.add_argument('--threshold_percentile', type=int, default=90, help='阈值超标技能得分的百分位阈值')

    args = parser.parse_args()

    print(f"加载预测: {args.pred_path}")
    print(f"加载真实: {args.true_path}")
    pred = np.load(args.pred_path)
    true = np.load(args.true_path)

    print(f"形状: pred={pred.shape}, true={true.shape}")

    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    metrics = compute_extreme_metrics(
        pred_flat, true_flat,
        percentile=args.percentile,
        top_k=args.top_k,
        threshold_percentile=args.threshold_percentile
    )
    print_metrics(metrics, "极端事件评估结果")


if __name__ == '__main__':
    main()
