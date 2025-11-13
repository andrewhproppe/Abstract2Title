from torch.utils.data import Dataset


class ArxivDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        input_ids, label_ids = self.tokenized_data[idx]
        return {
            "input_ids": input_ids.squeeze(),  # Remove extra dimensions if present
            "labels": label_ids.squeeze(),  # GPT-2 uses input and labels during training
        }


class AbstractTitleDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize the entire dataset here
        self.inputs = []
        self.labels = []
        for abstract, title in pairs:
            abstract_enc = tokenizer(
                abstract,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            title_enc = tokenizer(
                title,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.inputs.append(abstract_enc.input_ids.squeeze())  # Remove extra batch dimension
            self.labels.append(title_enc.input_ids.squeeze())  # Remove extra batch dimension

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}
