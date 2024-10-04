import torch
from transformers import BertTokenizer, BertModel, EncoderDecoderModel
from datasets import Dataset
from sklearn.model_selection import train_test_split
from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Check if a GPU is available
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

model_name = "bert-base-uncased"

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize encoder and decoder models separately
encoder = BertModel.from_pretrained(model_name)
decoder = BertModel.from_pretrained(model_name)  # Can be the same model or a different one

# Create the Encoder-Decoder model
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# Set the decoder_start_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id  # Typically, BERT uses [CLS] for this

# Fetch data
abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)

# Split the dataset
train_df, eval_df = train_test_split(papers, test_size=0.1)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Custom tokenize function for seq2seq
def tokenize_function(examples):
    # Tokenize inputs and outputs
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)

    # Combine the tokenized inputs and outputs
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],  # Use the output token IDs as labels
    }

# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    predict_with_generate=True,  # Important for seq2seq tasks
)

# Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Train the model
trainer.train()

# Generate titles from the abstracts
def generate_title(abstract):
    input_ids = tokenizer(abstract, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
    generated_ids = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    title = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return title

# Example usage
abstract_example = "Quantum computing is a rapidly evolving field of computer science that focuses on the development of computers..."
generated_title = generate_title(abstract_example)
print("Generated Title:", generated_title)