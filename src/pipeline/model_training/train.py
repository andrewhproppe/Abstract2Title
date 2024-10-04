import torch
from src.data.data import fetch_papers_for_training
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.to(device)

# GPT-2 needs padding tokens explicitly added, so we do this
tokenizer.pad_token = tokenizer.eos_token


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
                padding='max_length',
                return_tensors='pt'
            )
            title_enc = tokenizer(
                title,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.inputs.append(abstract_enc.input_ids.squeeze())  # Remove extra batch dimension
            self.labels.append(title_enc.input_ids.squeeze())  # Remove extra batch dimension

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.labels[idx]
        }


# Fetch abstract-title pairs from your data source
abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')

# Create dataset and dataloader
dataset = AbstractTitleDataset(abstract_title_pairs, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Save the model after training
model.save_pretrained('gpt2_finetuned')

# Load the fine-tuned model for testing
model = GPT2LMHeadModel.from_pretrained('gpt2_finetuned').to(device)
model.eval()  # Set model to evaluation mode


# Function to generate a title for a given abstract
def generate_title(abstract, max_length=50):
    inputs = tokenizer(abstract, return_tensors='pt').input_ids.to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title


# Example test: Generate a title for an abstract
test_abstract = "In this paper, we propose a new method for training deep neural networks with fewer parameters..."
generated_title = generate_title(test_abstract)
print(f"Generated Title: {generated_title}")
