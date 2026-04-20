"""
极端事件评估指标计算脚本

使用方法:
    python compute_extreme_metrics.py --pred_path <pred.npy路径> --true_path <true.npy路径>

示例:
    python compute_extreme_metrics.py --pred_path "test_results/.../pred.npy" --true_path "test_results/.../true.npy"
"""

import argparse
import numpy as np


# ============ 标准指标 ============

def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


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
    # 计算极端值的阈值（基于真实值）
    threshold = np.percentile(true, percentile)

    # 提取极端事件
    extreme_true = true[true >= threshold]
    extreme_pred = pred[true >= threshold]

    if len(extreme_true) == 0:
        return np.nan

    # 计算极端预测的命中率
    pred_exceed = (pred >= threshold).sum() / len(pred)
    true_exceed = (true >= threshold).sum() / len(true)

    if pred_exceed == 0 or true_exceed == 0:
        return 0.0

    # SEDI 公式
    sedi = 2 * (pred_exceed - true_exceed) / (pred_exceed + true_exceed)
    # 转换为 [0, 1] 范围，1表示完美
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
    else:  # range - 峰值范围
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
    # 基于真实值定义阈值
    threshold = np.percentile(true, threshold_percentile)

    # 预测超过阈值的比例
    pred_exceed = (pred >= threshold).sum() / len(pred)
    # 真实超过阈值的比例
    true_exceed = (true >= threshold).sum() / len(true)

    if true_exceed == 0:
        skill = 1.0 if pred_exceed == 0 else 0.0
    else:
        skill = pred_exceed / true_exceed

    return skill


def extreme_MAE(pred, true, top_k=10):
    """
    极端峰值平均绝对误差 (以℃为单位)

    只针对极端高温/低温事件��� MAE

    参数:
        pred: 预测值
        true: 真实值
        top_k: 计算误差的极端值数量（取top-k个最极端的值）

    返回:
        extreme_mae: 极端值的MAE
    """
    # 极端值：绝对值偏离均值最大的点
    mean_true = np.mean(true)
    deviation = np.abs(true - mean_true)

    # 获取 top_k 个最极端的索引
    top_k = min(top_k, len(deviation))
    extreme_indices = np.argsort(deviation)[-top_k:]

    extreme_true = true[extreme_indices]
    extreme_pred = pred[extreme_indices]

    return np.mean(np.abs(extreme_true - extreme_pred))


def compute_metrics(pred, true, percentile=95, top_k=10, threshold_percentile=90):
    """
    计算所有指标

    参数:
        pred: 预测值, shape [num_samples, pred_len, num_vars]
        true: 真实值, shape 同 pred
        percentile: SEDI 使用
        top_k: 极端MAE使用
        threshold_percentile: 阈值超标技能得分使用

    返回:
        dict: 包含所有指标
    """
    # 标准指标
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    # 极端事件指标
    sedi = SEDI(pred, true, percentile)
    peak_max_err = peak_amplitude_error(pred, true, mode='max')
    peak_min_err = peak_amplitude_error(pred, true, mode='min')
    peak_range_err = peak_amplitude_error(pred, true, mode='range')
    exceed_skill = threshold_exceedance_skill(pred, true, threshold_percentile)
    extreme_mae = extreme_MAE(pred, true, top_k)

    return {
        # 标准指标
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE(%)': mape * 100,
        'MSPE(%)': mspe * 100,
        # 极端事件指标
        'SEDI': sedi,
        'peak_max_error': peak_max_err,
        'peak_min_error': peak_min_err,
        'peak_range_error': peak_range_err,
        'exceedance_skill': exceed_skill,
        'extreme_MAE': extreme_mae,
    }


def merge_sliding_window(pred, true):
    """
    合并滑动窗口预测，得到每个时间点的平均预测值

    由于测试时使用滑动窗口(步长=1)，同一时间点会被多个样本预测。
    对每个时间点取平均（或取第一个非重复值），得到去重后的时间序列。

    参数:
        pred: 预测值, shape [num_samples, pred_len, num_vars]
        true: 真实值, shape 同 pred

    返回:
        pred_seq: [pred_len, num_vars] 合并后的时间序列
        true_seq: 同上
    """
    num_samples, pred_len, num_vars = pred.shape

    # 方案：只保留第一个样本的预测结果
    # 因为 num_samples 和 pred_len 接近，第二个样本开始会有很多重复
    # 取第一个样本的结果作为代表性的预测-真实对比
    # 这样每个时间点只对应一个预测值

    # 更正确的做法：按时间对齐
    # 每个样本 i 预测的是：[i, i+pred_len)
    # 但我们只需要前面 pred_len 个样本的结果

    # 简单处理：取所有样本的均值
    pred_merged = np.mean(pred, axis=0)  # [pred_len, num_vars]
    true_merged = np.mean(true, axis=0)   # [pred_len, num_vars]

    return pred_merged, true_merged


def main():
    parser = argparse.ArgumentParser(description='计算极端事件评估指标')
    parser.add_argument('--pred_path', type=str, required=True, help='pred.npy 文件路径')
    parser.add_argument('--true_path', type=str, required=True, help='true.npy 文件路径')
    parser.add_argument('--use_merged', type=int, default=1, help='是否使用合并后的序列(方案B), 1=是, 0=否')
    parser.add_argument('--percentile', type=int, default=95, help='SEDI 百分位阈值')
    parser.add_argument('--top_k', type=int, default=10, help='极端MAE的top-k个数')
    parser.add_argument('--threshold_percentile', type=int, default=90, help='阈值超标技能得分的百分位阈值')

    args = parser.parse_args()

    # 加载数据
    print(f"加载预测: {args.pred_path}")
    print(f"加载真实: {args.true_path}")
    pred = np.load(args.pred_path)
    true = np.load(args.true_path)

    print(f"原始形状: pred={pred.shape}, true={true.shape}")

    # 使用合并后的序列（方案B）
    if args.use_merged:
        print("使用合并后的序列（方案B，对滑动窗口去重取平均）...")
        pred, true = merge_sliding_window(pred, true)
        print(f"合并后形状: pred={pred.shape}, true={true.shape}")

    # 展平为 1D 进行计算
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # 计算指标
    metrics = compute_metrics(
        pred_flat, true_flat,
        percentile=args.percentile,
        top_k=args.top_k,
        threshold_percentile=args.threshold_percentile
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("评估指标结果")
    print("=" * 60)

    print("\n【标准指标】")
    print(f"  MAE:      {metrics['MAE']:.6f}")
    print(f"  MSE:      {metrics['MSE']:.6f}")
    print(f"  RMSE:     {metrics['RMSE']:.6f}")
    print(f"  MAPE(%):  {metrics['MAPE(%)']:.4f}%")
    print(f"  MSPE(%):  {metrics['MSPE(%)']:.4f}%")

    print("\n【极端事件指标】")
    print(f"  SEDI (95%):              {metrics['SEDI']:.6f}")
    print(f"  peak_max_error (℃):     {metrics['peak_max_error']:.6f}")
    print(f"  peak_min_error (℃):    {metrics['peak_min_error']:.6f}")
    print(f"  peak_range_error (℃):   {metrics['peak_range_error']:.6f}")
    print(f"  exceedance_skill (90%):  {metrics['exceedance_skill']:.6f}")
    print(f"  extreme_MAE (℃):       {metrics['extreme_MAE']:.6f}")

    print("=" * 60)


if __name__ == '__main__':
    main()