import torch
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-base"  # You can also use "facebook/bart-base" for a smaller version
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Check if a GPU is available
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Fetch data
abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(papers)

# Define a preprocessing function for tokenizing the input and target text
def preprocess_data(examples):
    inputs = [abstract for abstract in examples["input_text"]]
    targets = [title for title in examples["target_text"]]

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize targets with labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply the preprocessing function to the entire dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",           # Output directory
    evaluation_strategy="epoch",      # Evaluate at the end of each epoch
    learning_rate=2e-5,               # Adjust as needed
    per_device_train_batch_size=4,    # Batch size for training
    per_device_eval_batch_size=4,     # Batch size for evaluation
    num_train_epochs=3,               # Number of epochs
    weight_decay=0.01,                # Strength of weight decay
    logging_dir="./logs",             # Directory for storing logs
    logging_steps=10,
    save_steps=500,
    save_total_limit=2                # Save the last 2 checkpoints only
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # You may split your data for evaluation
)

# Train the model
trainer.train()

# Sample input text (an abstract)
input_text = papers['input_text'][0]

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate title
output_ids = model.generate(inputs["input_ids"].to(model.device), max_length=128, num_beams=4, early_stopping=True)
generated_title = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Title:", generated_title)