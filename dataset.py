from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pickle
import torch
from torch.utils.data import random_split
import random
from random import sample
from tqdm import tqdm

import numpy as np
try:
    from sklearn.preprocessing import MinMaxScaler
except:
    print("import wrong")


class TrainData(Dataset):

    def __init__(self, file_path, test_path, window_length=100,split=4,mask_ratio=0.5):
        self.data = pickle.load(
            open(file_path, "rb")
        )
        length = self.data.shape[0]

        self.mask_ratio = mask_ratio
        self.test_data = pickle.load(
            open(test_path, "rb")
        )
        self.data = np.concatenate([self.data, self.test_data])
        self.data = torch.Tensor(self.data)
        # 为了避免高斯噪声造成的影响过大，此处将原有的数值全部乘以20
        self.data = self.data[:length, :] * 20
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.data) - 100))
        self.split = split


    def get_mask(self, observed_mask, strategy_type):
        mask = torch.zeros_like(observed_mask)
        length = observed_mask.shape[0]
        if strategy_type == 0:
            # mask_ratio = self.mask_ratio

            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 == 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1
        else:
            # mask_ratio = 1 - self.mask_ratio
            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 != 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1

        return mask


    def __len__(self):
        return len(self.begin_indexes)

    def __getitem__(self, item):
        if random.random() < 0.5:
            strategy_type = 0
        else:
            strategy_type = 1

        observed_data = self.data[
            self.begin_indexes[item] :
               self.begin_indexes[item] + self.window_length
        ]
        observed_mask = torch.ones_like(observed_data)
        gt_mask = self.get_mask(observed_mask, strategy_type)
        timepoints = np.arange(self.window_length)
        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "strategy_type": strategy_type
        }

class TestData(Dataset):

    def __init__(self, file_path,label_path, train_path,window_length=100, get_label=False,window_split=1,strategy = 1,split=4,mask_list = []):
        self.strategy = strategy
        self.get_label = get_label
        self.data = pickle.load(
            open(file_path, "rb")
        )
        self.mask_list = mask_list
        length = self.data.shape[0]
        try:
            self.train_data  = pickle.load(
                open(train_path, "rb")
            )
        except:
            print("train data get wrong !")

        try:
            self.label = pickle.load(
                open(label_path,"rb")
            )
        except:
            print("label get wrong !")
        self.label = torch.LongTensor(self.label)
        self.data = np.concatenate([self.data, self.train_data])
        self.data = torch.Tensor(self.data)
        self.data = self.data[:length, :] * 20
        self.window_length = window_length
        self.begin_indexes = list(range(0, len(self.data) - 100, self.window_length // window_split))
        self.mask_index = 0
        self.split = split

    def __len__(self):
        return len(self.begin_indexes)

    def get_mask(self, observed_mask):
        mask = torch.zeros_like(observed_mask)

        length = observed_mask.shape[0]

        if self.strategy == 0:
            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 == 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1


        elif self.strategy == 1:
            skip = length // self.split
            for split_index, begin_index in enumerate(list(
                    range(0, length, skip)
            )):
                if split_index % 2 != 0:
                    mask[begin_index: min(begin_index + skip, length), :] = 1

        return mask




    def __getitem__(self, item):
        observed_data = self.data[
                        self.begin_indexes[item]:
                        self.begin_indexes[item] + self.window_length
                        ]
        observed_mask = torch.ones_like(observed_data)
        # print(f"item is {item}")
        gt_mask = self.get_mask(observed_mask)
        timepoints = np.arange(self.window_length)
        label = self.label[
            self.begin_indexes[item] :
           self.begin_indexes[item] + self.window_length
        ]


        if self.get_label:
            return {
                "observed_data": observed_data,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
                "label": label,
                'strategy_type': self.strategy
            }
        else:
            return {
                "observed_data": observed_data,
                "observed_mask": observed_mask,
                "gt_mask": gt_mask,
                "timepoints": timepoints,
                'strategy_type': self.strategy
            }

def get_mask(observed_mask, mask_ratio):
    mask = torch.zeros_like(observed_mask)

    original_mask_shape = mask.shape

    mask = mask.reshape(-1)
    total_index_list = list(range(len(mask)))

    selected_number = int(len(total_index_list) * mask_ratio)

    selected_index = sample(total_index_list, selected_number)

    selected_index = torch.LongTensor(selected_index)

    mask[selected_index] = 1

    mask = mask.reshape(original_mask_shape)

    return mask

def get_dataloader(train_path, test_path, label_path,batch_size = 32,window_split=1,split=4,mask_ratio=0.5):
    train_data = TrainData(train_path,test_path,split=split,mask_ratio=mask_ratio)
    train_data, valid_data = random_split(
        train_data, [len(train_data) - int(0.05 * len(train_data)) , int(0.05 * len(train_data)) ]
    )

    temp_dict = train_data.__getitem__(0)
    observed_mask = temp_dict['observed_mask']

    mask_list = []

    for i in tqdm(range(0,100)):
        mask_list.append(get_mask(observed_mask,mask_ratio=mask_ratio))


    test_data_strategy_1 = TestData(test_path, label_path, train_path,window_split=window_split,strategy=0,split=split,mask_list=mask_list)
    test_data_strategy_2 = TestData(test_path, label_path, train_path, window_split=window_split, strategy=1,split=split,mask_list=mask_list)

    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data,batch_size=batch_size,shuffle=True)

    test_loader1 = DataLoader(test_data_strategy_1,batch_size=batch_size)
    test_loader2 = DataLoader(test_data_strategy_2,batch_size=batch_size)

    return train_loader, valid_loader, test_loader1, test_loader2


if __name__ == "__main__":
    train_loader, valid_loader, test_loader1, test_loader2 = get_dataloader(
        "data/Machine/machine-1-1_train.pkl",
        "data/Machine/machine-1-1_test.pkl",
        "data/Machine/machine-1-1_test_label.pkl",
        split=8
    )
    for batch in test_loader2:
        break
    temp = batch["gt_mask"][23]
    temp1 = batch["gt_mask"][23]
    for item in temp:
        print(item)

    print("\n\n\n>>>>>>>>>>>\n\n\n")

    for batch in test_loader1:
        break
    temp = batch["gt_mask"][23]
    temp2 = batch["gt_mask"][23]
    for item in temp:
        print(item)

    print("check")

    print("and all zero")
    print(torch.any(temp1 * temp2)) # should be false

    print("or all one")
    print(torch.all(temp1 + temp2)) # should be true
