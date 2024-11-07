from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

# Add additional special tokens to explicitly mark sections
tokenizer.add_special_tokens({"additional_special_tokens": ["[ABSTRACT]", "[TITLE]"]})
model.resize_token_embeddings(len(tokenizer))  # Update model embeddings

def preprocess_data(examples):
    inputs = [f"[ABSTRACT] {abstract} [TITLE]" for abstract in examples['input_text']]
    targets = examples['target_text']

    # Combine and tokenize input with the target
    inputs_targets = [f"{input_text} {title}" for input_text, title in zip(inputs, targets)]

    # Tokenize with padding and truncation
    model_inputs = tokenizer(inputs_targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs

dataset = Dataset.from_pandas(papers)
tokenized_dataset = dataset.map(preprocess_data, batched=True)

from transformers import TrainerCallback

class PredictionPrintingCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, log_interval=10):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        # Only generate predictions every `log_interval` steps
        if state.global_step % self.log_interval == 0:
            # Pick a random sample from the dataset for prediction
            sample = self.dataset["input_text"][2]  # You can change this to pick a random one each time

            # Encode the input and generate predictions
            inputs = self.tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(args.device)

            # Generate output (adjust max_length for the length of the title)
            outputs = kwargs["model"].generate(input_ids, max_length=512, num_return_sequences=1)

            # Decode and print the prediction
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n[Step {state.global_step}] Prediction: {prediction}")

# Add the callback to the trainer
prediction_callback = PredictionPrintingCallback(tokenizer=tokenizer, dataset=papers, log_interval=10)

training_args = TrainingArguments(
    output_dir="./results_GPT2",           # Output directory
    save_strategy="epoch",
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch
    learning_rate=2e-5,                  # Adjust as needed
    per_device_train_batch_size=2,       # Batch size for training
    per_device_eval_batch_size=2,        # Batch size for evaluation
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
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Use a validation set if available
    callbacks=[prediction_callback]
)

# Fine-tune the model
trainer.train()