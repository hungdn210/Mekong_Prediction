import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_mekong_phase_a(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='Stung Treng.csv',
                 target='Value', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        self.args = args
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

        """
        df_raw.columns: ['Timestamp', 'Water.Level', 'Discharge.Daily', 'Rainfall.Manual']
        """

        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('Timestamp')
        # df_raw = df_raw[['Timestamp'] + [self.target]]

        # todo: check the length of test
        num_vali = int(0.1 * len(df_raw))
        num_test = int(0.2 * len(df_raw))
        num_train = len(df_raw) - num_test - num_vali
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['Timestamp']][border1:border2]
        df_stamp['Timestamp'] = pd.to_datetime(df_stamp['Timestamp'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['Timestamp'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['Timestamp'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['Timestamp'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['Timestamp'].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Timestamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise NotImplementedError

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # water_level, water_discharge, rainfall
        wl_x = self.data_x[:, 0:1][s_begin:s_end]
        wl_y = self.data_y[:, 0:1][r_begin:r_end]
        wd_x = self.data_x[:, 1:2][s_begin:s_end]
        rf_x = self.data_x[:, 2:3][s_begin:s_end]

        if pd.isna(wl_x).any():
            wl_x = None
        if pd.isna(wd_x).any():
            wd_x = None
        if pd.isna(rf_x).any():
            rf_x = None
        return wl_x, wd_x, rf_x, wl_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_mekong(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='Stung Treng.csv',
                 target='Value', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        """
        self.train_start_date = '1992-09-01T00:00Z'
        self.train_end_date = '2016-08-31T00:00Z'
        self.vali_start_date = '2016-09-01T00:00Z'
        self.vali_end_date = '2020-08-31T00:00Z'
        self.test_start_date = '2020-09-01T00:00Z'
        self.test_end_date = '2023-08-31T00:00Z'
        """
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
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('Timestamp')
        df_raw = df_raw[['Timestamp'] + [self.target]]

        """
        Training Range: 87660, 24 years, 1992-09-01 00:00:00+00:00 to 2016-08-31 00:00:00+00:00
        Validation Range: 14610, 4 years, 2016-09-01 00:00:00+00:00 to 2020-08-31 00:00:00+00:00
        Test Range: 10950, 3 years, 2020-09-01 00:00:00+00:00 to 2023-08-31 00:00:00+00:00
        """
        num_vali = 1461
        num_test = 1095
        num_train = 8766
        # num_train = len(df_raw) - num_test - num_vali
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

        df_stamp = df_raw[['Timestamp']][border1:border2]
        df_stamp['Timestamp'] = pd.to_datetime(df_stamp['Timestamp'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['Timestamp'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['Timestamp'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['Timestamp'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['Timestamp'].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Timestamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

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


class Dataset_mekong_cross(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data1_path='Stung Treng.csv',
                 data2_path='Kratie.csv',
                 target='Value', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        """
        self.train_start_date = '1992-09-01T00:00Z'
        self.train_end_date = '2016-08-31T00:00Z'
        self.vali_start_date = '2016-09-01T00:00Z'
        self.vali_end_date = '2020-08-31T00:00Z'
        self.test_start_date = '2020-09-01T00:00Z'
        self.test_end_date = '2023-08-31T00:00Z'
        """
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
        self.data1_path = data1_path
        self.data2_path = data2_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw1 = pd.read_csv(os.path.join(self.root_path, self.data1_path))
        df_raw2 = pd.read_csv(os.path.join(self.root_path, self.data2_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('Timestamp')
        df_raw1 = df_raw1[['Timestamp'] + [self.target]]
        df_raw2 = df_raw2[['Timestamp'] + [self.target]]

        """
        Training Range: 87660, 24 years, 1992-09-01 00:00:00+00:00 to 2016-08-31 00:00:00+00:00
        Validation Range: 14610, 4 years, 2016-09-01 00:00:00+00:00 to 2020-08-31 00:00:00+00:00
        Test Range: 10950, 3 years, 2020-09-01 00:00:00+00:00 to 2023-08-31 00:00:00+00:00
        """
        num_vali = 1461
        num_test = 1095
        num_train = 8766
        # num_train = len(df_raw1) - num_test - num_vali
        border1s = [0, num_train - self.seq_len, len(df_raw1) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw1)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw1.columns[1:]
            df_data1 = df_raw1[cols_data]
            df_data2 = df_raw2[cols_data]
        elif self.features == 'S':
            df_data1 = df_raw1[[self.target]]
            df_data2 = df_raw2[[self.target]]

        if self.scale:
            train_data = df_data1[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data1 = self.scaler.transform(df_data1.values)
            data2 = self.scaler.transform(df_data2.values)
        else:
            data1 = df_data1.values
            data2 = df_data2.values

        df_stamp = df_raw1[['Timestamp']][border1:border2]
        df_stamp['Timestamp'] = pd.to_datetime(df_stamp['Timestamp'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['Timestamp'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['Timestamp'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['Timestamp'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['Timestamp'].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Timestamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data1_x = data1[border1:border2]
        self.data1_y = data1[border1:border2]
        self.data2_x = data2[border1:border2]
        self.data2_y = data2[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data1_x[s_begin:s_end]
        seq_y = self.data2_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data1_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
