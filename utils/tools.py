import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# def visual(true, preds=None, name='./pic/test.pdf'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     if preds is not None:
#         plt.plot(preds, label='Prediction', linewidth=2)
#     plt.plot(true, label='GroundTruth', linewidth=2)
#     plt.legend()
#     plt.savefig(name, bbox_inches='tight')


def visual(true, preds=None, name='./pic/test.pdf', known_length=None):
    """
    Results visualization with shaded area for known data
    
    Args:
        true: 真实值（包含历史+未来）
        preds: 预测值（包含历史+未来）
        name: 保存路径
        known_length: 已知数据的长度（用于添加阴影）
    """
    plt.figure(figsize=(14, 6))
    
    # 添加阴影区域表示已知部分 - 使用更深的颜色
    if known_length is not None:
        plt.axvspan(0, known_length-1, alpha=0.3, color='gray', 
                   label='Historical Data', zorder=0)
    
    # 绘制预测线
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, color='red', alpha=0.8)
    
    # 绘制真实值线
    plt.plot(true, label='GroundTruth', linewidth=2, color='blue', alpha=0.8)
    
    # 添加分割线表示预测开始点
    if known_length is not None:
        plt.axvline(x=known_length-1, color='black', linestyle='--', 
                   alpha=0.6, linewidth=1.5, label='Forecast Start')
    
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Steps', fontsize=11)
    plt.ylabel('Precipitation (mm)', fontsize=11)
    plt.title('Time Series Forecasting Results', fontsize=12, fontweight='bold')
    
    # 美化图形
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(0.8)
    plt.gca().spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.close()  # 关闭图形以释放内存


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
