import torch

class Data(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row.to_dict()
    def __len__(self):
        return self.df.shape[0]