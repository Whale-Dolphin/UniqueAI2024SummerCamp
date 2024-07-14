from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][1:]
        y = self.data[idx][0].unsqueeze(0)
        return x, y