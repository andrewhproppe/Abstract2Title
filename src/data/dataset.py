import torch
from torch.utils.data import Dataset

class ArxivDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        input_ids, label_ids = self.tokenized_data[idx]
        return {
            'input_ids': input_ids.squeeze(),  # Remove extra dimensions if present
            'labels': label_ids.squeeze()      # GPT-2 uses input and labels during training
        }