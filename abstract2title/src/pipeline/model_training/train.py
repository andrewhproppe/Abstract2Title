import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer

from abstract2title.data.data import fetch_papers_for_training
from abstract2title.data.dataset import AbstractTitleDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-name",
        type=str,
        required=True,
        help="Name of the training dataset.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    args = parser.parse_args()

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)

    # GPT-2 needs padding tokens explicitly added, so we do this
    tokenizer.pad_token = tokenizer.eos_token

    # Fetch abstract-title pairs from your data source
    abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name=args.db_name)

    # Create dataset and dataloader
    dataset = AbstractTitleDataset(abstract_title_pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model after training
    model.save_pretrained("gpt2_finetuned")

    # Load the fine-tuned model for testing
    model = GPT2LMHeadModel.from_pretrained("gpt2_finetuned").to(device)
    model.eval()  # Set model to evaluation mode

    # Function to generate a title for a given abstract
    def generate_title(abstract, max_length=50):
        inputs = tokenizer(abstract, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            inputs, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True
        )
        title = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return title

    # Example test: Generate a title for an abstract
    test_abstract = "In this paper, we propose a new method for training deep neural networks with fewer parameters..."
    generated_title = generate_title(test_abstract)
    print(f"Generated Title: {generated_title}")


if __name__ == "__main__":
    main()
