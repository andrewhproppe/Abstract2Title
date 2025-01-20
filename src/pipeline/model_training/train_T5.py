import torch
import os

from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from src.data.data import load_dataset
from src.pipeline.model_training.utils import PredictionCallback

def preprocess_data(examples):
    inputs = ["summarize: " + text for text in examples["abstract"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize the targets with padding for batch processing
    labels = tokenizer(examples["title"], max_length=128, truncation=True, padding="max_length")

    # Set the labels in the input dictionary, replacing padding token ids with -100
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in l] for l in labels["input_ids"]
    ]
    return model_inputs

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "t5-base"  # You can also use "t5-base" or "t5-small" for smaller versions
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

papers = load_dataset('arxiv_papers_100000n_None_cat.csv', n_rows=100000)

dataset = Dataset.from_pandas(papers)
tokenized_dataset = dataset.map(preprocess_data, batched=True)

train_test_valid = tokenized_dataset.train_test_split(test_size=0.2)
valid_test_split = train_test_valid["test"].train_test_split(test_size=0.5)
train_dataset = train_test_valid["train"]
val_dataset = valid_test_split["train"]
test_dataset = valid_test_split["test"]

training_args = TrainingArguments(
    output_dir="./results_T5",           # Output directory
    save_strategy="epoch",
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch
    learning_rate=2e-5,                  # Adjust as needed
    per_device_train_batch_size=4,       # Batch size for training
    per_device_eval_batch_size=4,        # Batch size for evaluation
    num_train_epochs=2,                  # Number of epochs
    weight_decay=0.01,                   # Strength of weight decay
    logging_dir="./logs",                # Directory for storing logs
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,                  # Save the last 2 checkpoints only
    report_to="none",
)

prediction_callback = PredictionCallback(
    model=model,
    tokenizer=tokenizer,
    dataset=val_dataset,
    log_interval=100,
    prepend_text="summarize: ",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[prediction_callback],
)

trainer.train()