import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
# from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# class Dataset_M4(Dataset):
#     def __init__(self, args, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
#                  seasonal_patterns='Yearly'):
#         # size [seq_len, label_len, pred_len]
#         # init
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.root_path = root_path

#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]

#         self.seasonal_patterns = seasonal_patterns
#         self.history_size = M4Meta.history_size[seasonal_patterns]
#         self.window_sampling_limit = int(self.history_size * self.pred_len)
#         self.flag = flag

#         self.__read_data__()

#     def __read_data__(self):
#         # M4Dataset.initialize()
#         if self.flag == 'train':
#             dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
#         else:
#             dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
#         training_values = np.array(
#             [v[~np.isnan(v)] for v in
#              dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
#         self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
#         self.timeseries = [ts for ts in training_values]

#     def __getitem__(self, index):
#         insample = np.zeros((self.seq_len, 1))
#         insample_mask = np.zeros((self.seq_len, 1))
#         outsample = np.zeros((self.pred_len + self.label_len, 1))
#         outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

#         sampled_timeseries = self.timeseries[index]
#         cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
#                                       high=len(sampled_timeseries),
#                                       size=1)[0]

#         insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
#         insample[-len(insample_window):, 0] = insample_window
#         insample_mask[-len(insample_window):, 0] = 1.0
#         outsample_window = sampled_timeseries[
#                            max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
#         outsample[:len(outsample_window), 0] = outsample_window
#         outsample_mask[:len(outsample_window), 0] = 1.0
#         return insample, outsample, insample_mask, outsample_mask

#     def __len__(self):
#         return len(self.timeseries)

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

#     def last_insample_window(self):
#         """
#         The last window of insample size of all timeseries.
#         This function does not support batching and does not reshuffle timeseries.

#         :return: Last insample window of all timeseries. Shape "timeseries, insample size"
#         """
#         insample = np.zeros((len(self.timeseries), self.seq_len))
#         insample_mask = np.zeros((len(self.timeseries), self.seq_len))
#         for i, ts in enumerate(self.timeseries):
#             ts_last_window = ts[-self.seq_len:]
#             insample[i, -len(ts):] = ts_last_window
#             insample_mask[i, -len(ts):] = 1.0
#         return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class Dataset_Texas_Freeze(Dataset):
    def __init__(self,args, root_path, flag='train', size=None,
                 features='S', data_path='Texas_Freeze.csv',
                 target='OT', scale=True, timeenc=0, freq='h',seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]

        # 存储args参数，即使暂时不使用它
        self.args = args

        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 读取完整数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 确保日期列为datetime格式
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 定义数据集划分的日期边界
        # 训练集：2017-01-01 到 2019-12-31（3年）
        # 验证集：2020-01-01 到 2021-01-31（1年1个月）
        # 测试集：2021-02-05 到 2021-02-17（剩余时间）
        train_end_date = pd.to_datetime('2019-12-31 23:00:00')
        val_start_date = pd.to_datetime('2020-01-01 00:00:00')
        val_end_date = pd.to_datetime('2021-01-31 23:00:00')
        test_start_date = pd.to_datetime('2021-02-05 00:00:00')
        test_end_date = pd.to_datetime('2021-02-17 23:00:00')
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 假设第一列是日期
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            
        # 标准化处理 - 首先获取训练数据用于拟合标准化器
        train_df = df_raw[df_raw['date'] <= train_end_date]
        if self.features == 'M' or self.features == 'MS':
            train_df_data = train_df[cols_data]
        elif self.features == 'S':
            train_df_data = train_df[[self.target]]
            
        if self.scale:
            self.scaler.fit(train_df_data.values)
        
        # 根据标志划分数据
        if self.flag == 'train':
            subset = df_raw[df_raw['date'] <= train_end_date]
            
        elif self.flag == 'val':
            # 验证集需要额外的seq_len长度作为历史数据
            val_main = df_raw[(df_raw['date'] >= val_start_date) & 
                             (df_raw['date'] <= val_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = val_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < val_start_date)]
            
            # 组合历史数据和验证集数据
            subset = pd.concat([history_data, val_main])
            
        elif self.flag == 'test':
            # 测试集需要额外的seq_len长度作为历史数据
            test_main = df_raw[(df_raw['date'] >= test_start_date) & 
                             (df_raw['date'] <= test_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = test_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < test_start_date)]
            
            # 组合历史数据和测试集数据
            subset = pd.concat([history_data, test_main])
        
        # 根据选择的子集重构特征数据
        if self.features == 'M' or self.features == 'MS':
            data_subset = subset[cols_data]
        elif self.features == 'S':
            data_subset = subset[[self.target]]
            
        # 对选定的子集应用标准化
        if self.scale:
            data = self.scaler.transform(data_subset.values)
        else:
            data = data_subset.values
            
        # 时间特征处理
        df_stamp = subset[['date']]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        # 设置数据
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        # 对所有集合使用相同的逻辑
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # 对所有集合使用相同的逻辑
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    




class Dataset_Northwest_Heatwave(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='Texas_Freeze.csv',
                 target='OT', scale=True, timeenc=0, freq='h',seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]

        # 存储args参数，即使暂时不使用它
        self.args = args

        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 读取完整数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 确保日期列为datetime格式
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 定义数据集划分的日期边界
        train_end_date = pd.to_datetime('2019-12-31 23:00:00')
        val_start_date = pd.to_datetime('2020-01-01 00:00:00')
        # val_end_date = pd.to_datetime('2021-06-20 23:00:00')
        val_end_date = pd.to_datetime('2021-06-10 23:00:00')
        test_start_date = pd.to_datetime('2021-06-15 00:00:00')
        test_end_date = pd.to_datetime('2021-06-30 23:00:00')
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 假设第一列是日期
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            
        # 标准化处理 - 首先获取训练数据用于拟合标准化器
        train_df = df_raw[df_raw['date'] <= train_end_date]
        if self.features == 'M' or self.features == 'MS':
            train_df_data = train_df[cols_data]
        elif self.features == 'S':
            train_df_data = train_df[[self.target]]
            
        if self.scale:
            self.scaler.fit(train_df_data.values)
        
        # 根据标志划分数据
        if self.flag == 'train':
            subset = df_raw[df_raw['date'] <= train_end_date]
            
        elif self.flag == 'val':
            # 验证集需要额外的seq_len长度作为历史数据
            val_main = df_raw[(df_raw['date'] >= val_start_date) & 
                             (df_raw['date'] <= val_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = val_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < val_start_date)]
            
            # 组合历史数据和验证集数据
            subset = pd.concat([history_data, val_main])
            
        elif self.flag == 'test':
            # 测试集需要额外的seq_len长度作为历史数据
            test_main = df_raw[(df_raw['date'] >= test_start_date) & 
                             (df_raw['date'] <= test_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = test_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < test_start_date)]
            
            # 组合历史数据和测试集数据
            subset = pd.concat([history_data, test_main])
        
        # 根据选择的子集重构特征数据
        if self.features == 'M' or self.features == 'MS':
            data_subset = subset[cols_data]
        elif self.features == 'S':
            data_subset = subset[[self.target]]
            
        # 对选定的子集应用标准化
        if self.scale:
            data = self.scaler.transform(data_subset.values)
        else:
            data = data_subset.values
            
        # 时间特征处理
        df_stamp = subset[['date']]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        # 设置数据
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        # 对所有集合使用相同的逻辑
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # 对所有集合使用相同的逻辑
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class Dataset_PEMS(Dataset):
    def __init__(self, args,root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',seasonal_patterns=None):
        
        # 存储args参数，即使暂时不使用它
        self.args = args
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # 将1维时间特征改为4维
        seq_x_mark = torch.zeros((seq_x.shape[0], 4))
        seq_y_mark = torch.zeros((seq_x.shape[0], 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Zhengzhou_Precipitation(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='zhengzhou_precipitation_timeseries.csv',
                 target='feature1', scale=True, timeenc=0, freq='30min', seasonal_patterns=None):
        
        self.args = args
        
        # 存储args参数，即使暂时不使用它
        self.args = args

        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 读取完整数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 确保日期列为datetime格式
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 定义数据集划分的日期边界
        train_end_date = pd.to_datetime('2019-12-31 18:00:00')
        val_start_date = pd.to_datetime('2020-01-01 00:00:00')
        val_end_date = pd.to_datetime('2021-06-30 18:00:00')
        test_start_date = pd.to_datetime('2021-07-01 00:00:00')
        test_end_date = pd.to_datetime('2021-07-23 18:00:00')
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 假设第一列是日期
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            
        # 标准化处理 - 首先获取训练数据用于拟合标准化器
        train_df = df_raw[df_raw['date'] <= train_end_date]
        if self.features == 'M' or self.features == 'MS':
            train_df_data = train_df[cols_data]
        elif self.features == 'S':
            train_df_data = train_df[[self.target]]
            
        if self.scale:
            self.scaler.fit(train_df_data.values)
        
        # 根据标志划分数据
        if self.flag == 'train':
            subset = df_raw[df_raw['date'] <= train_end_date]
            
        elif self.flag == 'val':
            # 验证集需要额外的seq_len长度作为历史数据
            val_main = df_raw[(df_raw['date'] >= val_start_date) & 
                             (df_raw['date'] <= val_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = val_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < val_start_date)]
            
            # 组合历史数据和验证集数据
            subset = pd.concat([history_data, val_main])
            
        elif self.flag == 'test':
            # 测试集需要额外的seq_len长度作为历史数据
            test_main = df_raw[(df_raw['date'] >= test_start_date) & 
                             (df_raw['date'] <= test_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = test_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < test_start_date)]
            
            # 组合历史数据和测试集数据
            subset = pd.concat([history_data, test_main])
        
        # 根据选择的子集重构特征数据
        if self.features == 'M' or self.features == 'MS':
            data_subset = subset[cols_data]
        elif self.features == 'S':
            data_subset = subset[[self.target]]
            
        # 对选定的子集应用标准化
        if self.scale:
            data = self.scaler.transform(data_subset.values)
        else:
            data = data_subset.values
            
        # 时间特征处理
        df_stamp = subset[['date']]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        # 设置数据
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        # 对所有集合使用相同的逻辑
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # 对所有集合使用相同的逻辑
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    



class Dataset_Italy_Heatwave(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='Texas_Freeze.csv',
                 target='OT', scale=True, timeenc=0, freq='h',seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]

        # 存储args参数，即使暂时不使用它
        self.args = args

        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 读取完整数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 确保日期列为datetime格式
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 定义数据集划分的日期边界
        train_end_date = pd.to_datetime('2019-12-31 23:00:00')
        val_start_date = pd.to_datetime('2020-01-01 00:00:00')
        val_end_date = pd.to_datetime('2021-06-30 23:00:00')
        test_start_date = pd.to_datetime('2021-07-01 00:00:00')
        test_end_date = pd.to_datetime('2021-07-31 23:00:00')
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 假设第一列是日期
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            
        # 标准化处理 - 首先获取训练数据用于拟合标准化器
        train_df = df_raw[df_raw['date'] <= train_end_date]
        if self.features == 'M' or self.features == 'MS':
            train_df_data = train_df[cols_data]
        elif self.features == 'S':
            train_df_data = train_df[[self.target]]
            
        if self.scale:
            self.scaler.fit(train_df_data.values)
        
        # 根据标志划分数据
        if self.flag == 'train':
            subset = df_raw[df_raw['date'] <= train_end_date]
            
        elif self.flag == 'val':
            # 验证集需要额外的seq_len长度作为历史数据
            val_main = df_raw[(df_raw['date'] >= val_start_date) & 
                             (df_raw['date'] <= val_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = val_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < val_start_date)]
            
            # 组合历史数据和验证集数据
            subset = pd.concat([history_data, val_main])
            
        elif self.flag == 'test':
            # 测试集需要额外的seq_len长度作为历史数据
            test_main = df_raw[(df_raw['date'] >= test_start_date) & 
                             (df_raw['date'] <= test_end_date)]
            
            # 额外获取seq_len长度的历史数据
            history_start_date = test_start_date - pd.Timedelta(hours=self.seq_len)
            history_data = df_raw[(df_raw['date'] >= history_start_date) & 
                                (df_raw['date'] < test_start_date)]
            
            # 组合历史数据和测试集数据
            subset = pd.concat([history_data, test_main])
        
        # 根据选择的子集重构特征数据
        if self.features == 'M' or self.features == 'MS':
            data_subset = subset[cols_data]
        elif self.features == 'S':
            data_subset = subset[[self.target]]
            
        # 对选定的子集应用标准化
        if self.scale:
            data = self.scaler.transform(data_subset.values)
        else:
            data = data_subset.values
            
        # 时间特征处理
        df_stamp = subset[['date']]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        # 设置数据
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        # 对所有集合使用相同的逻辑
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # 对所有集合使用相同的逻辑
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)





class Dataset_Custom_Weather(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
