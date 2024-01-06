import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def data_load(path="data/df_train.parq"):
    os.system(f"dvc pull {path}")
    df = pd.read_parquet(path)
    return df


class DataDigits(Dataset):
    def __init__(self, df, target_label="target"):
        data = df.drop(target_label, axis=1).values
        target = df[target_label].values
        data_numpy = np.array([[j for j in i] for i in data])
        self.data = torch.FloatTensor(data_numpy)
        self.target = torch.LongTensor(target)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.target)
