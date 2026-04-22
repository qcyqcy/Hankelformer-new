from data_provider.data_loader import Dataset_Custom, Dataset_Texas_Freeze, Dataset_Northwest_Heatwave, Dataset_Antarctic_Heat, Dataset_PEMS
from torch.utils.data import DataLoader
import torch
import numpy as np

# 获取主脚本设置的种子，默认2023
_worker_seed = 2023

def _worker_init_fn(worker_id):
    """为每个 DataLoader worker 设置相同的种子，确保可复现性"""
    np.random.seed(_worker_seed + worker_id)
    # 确保每个 worker 有确定性的随机状态
    worker_seed = _worker_seed + worker_id
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

data_dict = {
    'custom': Dataset_Custom,
    'Texas_Freeze': Dataset_Texas_Freeze,
    'Northwest_Heatwave': Dataset_Northwest_Heatwave,
    'Antarctic_Heat': Dataset_Antarctic_Heat,
    'PEMS': Dataset_PEMS,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn,  # 确保 worker 种子一致性
        persistent_workers=False if args.num_workers == 0 else True,  # 保持 worker 进程
        pin_memory=True  # 加速数据加载
    )
    return data_set, data_loader