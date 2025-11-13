import torch
import os

from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from src.data.data import load_dataset
from src.pipeline.model_training.utils import PredictionCallback

def preprocess_data(examples):
    inputs = [abstract for abstract in examples["abstract"]]
    targets = [title for title in examples["title"]]

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize targets with labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained BART model and tokxenizer
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

# papers = load_dataset('arxiv_papers_100000n_None_cat.csv', n_rows=50000)
papers = load_dataset('arxiv_papers_100000n_quant-ph_cat.csv', n_rows=50000)

dataset = Dataset.from_pandas(papers)
tokenized_dataset = dataset.map(preprocess_data, batched=True)

train_test_valid = tokenized_dataset.train_test_split(test_size=0.1)
valid_test_split = train_test_valid["test"].train_test_split(test_size=0.5)
train_dataset = train_test_valid["train"]
val_dataset = valid_test_split["train"]
test_dataset = valid_test_split["test"]

training_args = TrainingArguments(
    output_dir="./results_BART_quant-ph_cat",
    save_strategy="epoch",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_steps=2500,
    save_steps=100,
    save_total_limit=2,
    report_to="wandb",
    # report_to="none",
)

prediction_callback = PredictionCallback(
    model=model,
    tokenizer=tokenizer,
    dataset=val_dataset,
    log_interval=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[prediction_callback],
)

trainer.train()