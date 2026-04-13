import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

def add_gaussian_noise(x, std=0.01):
    """
    为输入数据添加高斯噪声
    Args:
        x: 输入张量 [batch_size, seq_len, input_size]
        std: 高斯噪声的标准差，默认为0.01
    Returns:
        添加噪声后的张量
    """
    noise = torch.randn_like(x) * std
    return x + noise


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 融合层
        self.fusion_layer = nn.Linear(configs.d_model * 2, configs.d_model)

        # 投影层
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # # 修改对比学习投影头
        # self.contrastive_head = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model, configs.d_model)  # 新的维度
        # )

    def encode(self, x, x_mark):
        # 嵌入和编码
        enc_out = self.enc_embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # 对整个序列进行全局平均池化
        global_repr = enc_out.mean(dim=1)  # [batch_size, d_model]
        
        # 对比学习投影
        # contrastive_repr = self.contrastive_head(global_repr)  # [batch_size, contrastive_dim]
        
        return enc_out, global_repr

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # 归一化
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # 通过添加高斯噪声处理x_enc
        x_enc_processed = add_gaussian_noise(x_enc, std=0.01)

        # 编码原始输入和增强样本
        enc_out, contrastive_repr = self.encode(x_enc, x_mark_enc)
        enc_out_processed, contrastive_repr_processed = self.encode(x_enc_processed, x_mark_enc)

        # # 打印形状
        # print("enc_out shape",enc_out.shape) # torch.Size([16, 325, 512])
        # print("enc_out_processed shape",enc_out_processed.shape) # torch.Size([16,
        # print("contrastive_repr shape",contrastive_repr.shape) # torch.Size([16, 325, 512])
        # print("contrastive_repr_processed shape",contrastive_repr_processed.shape) # torch.Size([16, 325, 512])

        # 融合两个编码
        fused_enc_out = self.fusion_layer(torch.cat([enc_out, enc_out_processed], dim=-1))

        # print("fused_enc_out shape",fused_enc_out.shape) # torch.Size([16, 325, 512])


        # 预测
        dec_out = self.projector(fused_enc_out).permute(0, 2, 1)[:, :, :x_enc.shape[2]]

        if self.use_norm:
            # 反归一化
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, contrastive_repr, contrastive_repr_processed

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, contrastive_repr, contrastive_repr_processed = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], contrastive_repr, contrastive_repr_processed