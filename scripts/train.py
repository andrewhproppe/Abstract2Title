from src.data.data import fetch_papers_for_training

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# GPT-2 needs padding tokens explicitly added, so we do this
tokenizer.pad_token = tokenizer.eos_token

def tokenize_pair(abstract, title, tokenizer, max_length=512):
    # Tokenize the abstract
    abstract_enc = tokenizer(
        abstract,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

    # Tokenize the title (prediction target)
    title_enc = tokenizer(
        title,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

    # GPT-2 does not have a "decoder," so we use the full input sequence (abstract + title)
    inputs = abstract_enc.input_ids
    labels = title_enc.input_ids

    return inputs, labels

from torch.utils.data import DataLoader, Dataset

class AbstractTitleDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        abstract, title = self.pairs[idx]
        inputs, labels = tokenize_pair(abstract, title, self.tokenizer, self.max_length)
        return {
            'input_ids': inputs.squeeze(),
            'labels': labels.squeeze()
        }

# # Example pairs of abstracts and titles
# abstract_title_pairs = [
#     ("Abstract text 1...", "Title 1"),
#     ("Abstract text 2...", "Title 2"),
#     # Add more pairs
# ]

abstract_title_pairs = fetch_papers_for_training(limit=100, db_name='arxiv_papers_100n.db')

# Create dataset and dataloader
dataset = AbstractTitleDataset(abstract_title_pairs, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

from transformers import AdamW

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch['input_ids']
        labels = batch['labels']

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')