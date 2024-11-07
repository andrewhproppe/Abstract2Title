import torch
import os

from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
from src.pipeline.model_training.utils import PredictionCallback

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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

# Fetch data
abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(papers)

# Apply the preprocessing function to the entire dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Split into train, validation, and test sets
train_test_valid = tokenized_dataset.train_test_split(test_size=0.2)
valid_test_split = train_test_valid["test"].train_test_split(test_size=0.5)
train_dataset = train_test_valid["train"]
val_dataset = valid_test_split["train"]
test_dataset = valid_test_split["test"]

# Define the TrainingArguments
training_args = TrainingArguments(
    output_dir="./results_BART",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

prediction_callback = PredictionCallback(
    model=model,
    tokenizer=tokenizer,
    dataset=val_dataset,
    log_interval=10
)

# Initialize Trainer with the callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[prediction_callback],
)

# Train the model
trainer.train()