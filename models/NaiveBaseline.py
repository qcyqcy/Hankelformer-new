import torch
import torch.nn as nn


class Model(nn.Module):
    """
    朴素下界模型：持续性预测（Persistence Forecast）

    取输入序列的最后一个时间步，重复 pred_len 次作为预测。
    不需要训练，is_training=0 模式下直接使用。
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """
        参数:
            batch_x: [batch, seq_len, vars]
            batch_x_mark: [batch, seq_len, time_dim]  (不使用)
            dec_inp: [batch, label_len+pred_len, vars] (不使用)
            batch_y_mark: [batch, label_len+pred_len, time_dim] (不使用)

        返回:
            naive_pred: [batch, pred_len, vars]
        """
        last_value = batch_x[:, -1:, :]                # [batch, 1, vars]
        pred_len = self.args.pred_len
        naive_pred = last_value.repeat(1, pred_len, 1)  # [batch, pred_len, vars]
        return naive_pred
