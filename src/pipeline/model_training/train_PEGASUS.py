from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from transformers import TrainerCallback
import random

# Load pre-trained PEGASUS model and tokenizer
model_name = "google/pegasus-xsum"  # or use a smaller domain-specific model if available
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def preprocess_data(examples):
    inputs = examples['input_text']  # Use the abstract as input
    targets = examples['target_text']  # Title as target

    # Tokenize with padding and truncation
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids

    model_inputs["labels"] = labels
    return model_inputs

dataset = Dataset.from_pandas(papers)
tokenized_dataset = dataset.map(preprocess_data, batched=True)


class TitlePredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, log_interval=10):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        # Only generate predictions every `log_interval` steps
        if state.global_step % self.log_interval == 0:
            # Pick a random sample from the dataset for prediction
            sample_index = random.randint(0, len(self.dataset["input_text"]) - 1)
            sample = self.dataset["input_text"][sample_index]

            # Tokenize the sample abstract
            inputs = self.tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(args.device)

            # Generate a title prediction (adjust max_length as needed)
            outputs = kwargs["model"].generate(input_ids, max_length=30, num_beams=4, early_stopping=True)

            # Decode the generated title and print it
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n[Step {state.global_step}] Predicted Title: {prediction}\n")
            # print(f"Original Abstract: {sample}\n")


# Add the callback to the trainer
prediction_callback = TitlePredictionCallback(tokenizer=tokenizer, dataset=papers, log_interval=10)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results_PEGASUS",      # Output directory
    save_strategy="epoch",
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch
    learning_rate=2e-5,                  # Adjust as needed
    per_device_train_batch_size=1,       # Batch size for training
    per_device_eval_batch_size=1,        # Batch size for evaluation
    num_train_epochs=3,                  # Number of epochs
    weight_decay=0.01,                   # Strength of weight decay
    logging_dir="./logs",                # Directory for storing logs
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,                  # Save the last 2 checkpoints only
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,#  ['train'],
    eval_dataset=tokenized_dataset, #['validation'],
    callbacks=[prediction_callback]  # Add the custom callback here
)

trainer.train()
