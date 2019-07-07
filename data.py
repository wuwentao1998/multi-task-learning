import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler
import argparse


class mergeData(object):
    def __init__(
            self,
            path,
            num_classes,
            batch_size,
            train_ratio,
            val_ratio,
            useless_columns,
            display_step,
            target_news,
            target_fin):

        self.path = path
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.useless_columns = useless_columns
        self.target_news = target_news
        self.target_fin = target_fin
        self.display_step = display_step

        self.data = self._load_data()
        self.max_steps = self._calculate_max_steps()
        self.input_dimension = self._calulate_input_dimension()
        self.num_samples = 0
        self.pre_compute_data = self._load_pre_compute()

        self.train_index = None
        self.val_index = None
        self.test_index = None

        self._split_data()

    def _load_data(self):
        data = pd.read_pickle(self.path)
        return data


    def _calculate_max_steps(self):
        return self.data.groupby(["stock_code"])["time"].nunique().max() - 1


    def _calulate_input_dimension(self):
        num_columns = len(list(self.data.columns))
        return num_columns - len(self.useless_columns) - 2


    def _pre_compute(self):
        stock_code_list = self.data["stock_code"].unique().tolist()

        column_list = list(self.data.columns)
        column_x_list = column_list.copy()
        useless_columns_plus = self.useless_columns.copy()
        useless_columns_plus.append(self.target_news)
        useless_columns_plus.append(self.target_fin)
        for column_i in useless_columns_plus:
            if column_i in column_x_list:
                column_x_list.remove(column_i)

        scaler = MinMaxScaler()
        self.data[column_x_list] = scaler.fit_transform(self.data[column_x_list])

        num_samples = int(self.data.shape[0])

        pre_compute_x = np.zeros([num_samples, self.max_steps, self.input_dimension])
        labels = np.zeros([num_samples, self.num_classes])
        seq_lens = np.zeros([num_samples], dtype = int)

        min_time_rank = self.data["time_rank"].min()
        max_time_rank = self.data["time_rank"].max()

        temp = 0
        for stock_code_i in stock_code_list:
            for rank_num in range(min_time_rank + 1, max_time_rank + 1):
                if self.data[(self.data["stock_code"] == stock_code_i) & (self.data["time_rank"] == rank_num)].shape[0] == 0:
                    continue

                rank_i = self.data[(self.data["stock_code"] == stock_code_i) & (
                        self.data["time_rank"] == rank_num)]["fin_rank"].tolist()[0]

                if rank_i <= 2:
                    continue

                pre_compute_x[temp, :rank_i - 2, :] = self.data[(self.data["stock_code"] == stock_code_i) &
                                                                  (self.data["time_rank"] < rank_num)].as_matrix(column_x_list)
                label_st = int(self.data[(self.data["stock_code"] == stock_code_i) & (self.data["time_rank"] == rank_num)]["st"].tolist()[0])
                label_sentiment = int(self.data[(self.data["stock_code"] == stock_code_i) & (self.data["time_rank"] == rank_num)]["mood"].tolist()[0])

                seq_lens[temp] = rank_i - 2

                if (label_st == 0) & (label_sentiment == 0):
                    labels[temp,:] = [1.0,0.0,1.0,0.0]
                elif (label_st == 1) & (label_sentiment == 0):
                    labels[temp,:] = [0.0,1.0,1.0,0.0]
                elif (label_st == 0) & (label_sentiment == 1):
                    labels[temp,:] = [1.0,0.0,0.0,1.0]
                else:
                    labels[temp,:] = [0.0,1.0,0.0,1.0]

                temp += 1



        pre_compute_data = {
            "x" : pre_compute_x,
            "y" : labels,
            "seq_len" : seq_lens
        }

        self.num_samples = temp
        return pre_compute_data


    def _load_pre_compute(self):
        pre_compute_dir = "./pre_compute/"
        filename = "data.npy"
        pre_compute_path = os.path.join(pre_compute_dir, filename)

        if not os.path.exists(pre_compute_dir):
            os.makedirs(pre_compute_dir)

        if os.path.exists(pre_compute_path):
            pre_compute_data = np.load(pre_compute_path)[()]
            self.num_samples = len(pre_compute_data["y"])
        else:
            pre_compute_data = self._pre_compute()
            np.save(pre_compute_path, pre_compute_data)
        return pre_compute_data


    def _split_data(self):
        train_size = int(self.train_ratio * self.num_samples)
        val_size = int(self.val_ratio * self.num_samples)
        self.train_index = random.sample(range(0, self.num_samples), train_size)
        val_test_index = [index_i for index_i in range(0, self.num_samples) if index_i not in self.train_index]
        self.val_index = random.sample(val_test_index, val_size)
        self.test_index = [index_i for index_i in val_test_index if index_i not in self.val_index]
        # self.pos_case_index = [index_i for index_i in self.train_index if
        #                        index_i in np.where(self.pre_compute_data["y"] == [0, 1])[0]]


    def _generate_batch(self, index):
        index_size = len(index)
        batch_x = np.zeros([index_size, self.max_steps, self.input_dimension])
        label = np.zeros([index_size, self.num_classes])
        seq_lens = np.zeros([index_size], dtype=int)

        temp = 0
        for item in index:
            batch_x[temp, :, :] = self.pre_compute_data["x"][item]
            label[temp, :] = self.pre_compute_data["y"][item]
            seq_lens[temp] = self.pre_compute_data["seq_len"][item]
            temp += 1

        return batch_x, label, seq_lens


    def _generate_balanced_training_index(self):
        num_pos_cases = int(self.batch_size * self.resample_ratio)

        if num_pos_cases > len(self.pos_case_index):
            num_pos_cases = len(self.pos_case_index)

        neg_case_index = [index_i for index_i in self.train_index if index_i not in self.pos_case_index]
        selected_neg_index = random.sample(neg_case_index, self.batch_size - num_pos_cases)
        selected_pos_index = random.sample(self.pos_case_index, num_pos_cases)

        return selected_neg_index + selected_pos_index


    def next_batch(self):
        # time_start = time.time()
        # if self.resample_training:
        #     index = self._generate_balanced_training_index()
        # else:
        index = random.sample(self.train_index, self.batch_size)
        index = np.array(index)
        batch_x, label, seq_lens = self._generate_batch(index)
        return batch_x, label, seq_lens


    def validation(self):
        index = np.array(self.val_index)
        batch_x, label, seq_lens = self._generate_batch(index)
        return batch_x, label, seq_lens


    def testing(self):
        index = np.array(self.test_index)
        batch_x, label, seq_lens = self._generate_batch(index)
        return batch_x, label, seq_lens