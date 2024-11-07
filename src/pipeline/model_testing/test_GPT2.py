import random
import torch

from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Fetch data
abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)
eval_df = papers.sample(frac=0.9, random_state=42)

# Load the trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("../model_training/results_GPT2/checkpoint-189")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to EOS if not done during training

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Special token to separate abstract and title in GPT-2
separator_token = "<|sep|>"

for _ in range(5):
    random_idx = random.randint(0, len(eval_df)-1)

    abstract = eval_df.iloc[random_idx]['input_text']
    true_title = eval_df.iloc[random_idx]['target_text']

    # Prepare input with separator token
    input_text = f"{abstract} {separator_token}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Generate predicted title
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=4,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id  # Ensure padding token ID is set
    )
    predicted_title = tokenizer.decode(output_ids[0], skip_special_tokens=True).split(separator_token)[-1].strip()

    print(f'True Title: {true_title}\n')
    print(f'Predicted Title: {predicted_title}\n')
    print(f'Abstract: {abstract}\n\n\n')
