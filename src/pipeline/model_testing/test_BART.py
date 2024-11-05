import random

from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments

abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)
eval_df = papers.sample(frac=0.9, random_state=42)

# raise RuntimeError
model = BartForConditionalGeneration.from_pretrained("../model_training/results_BART/checkpoint-96")
model_name = "facebook/bart-large"  # You can also use "facebook/bart-base" for a smaller version
tokenizer = BartTokenizer.from_pretrained(model_name)

for _ in range(5):
    random_idx = random.randint(0, len(eval_df)-1)

    abstract = eval_df.iloc[random_idx]['input_text']
    true_title = eval_df.iloc[random_idx]['target_text']

    inputs = tokenizer(abstract, return_tensors="pt", max_length=512, truncation=True)

    # Generate title
    output_ids = model.generate(inputs["input_ids"].to(model.device), max_length=128, num_beams=4, early_stopping=True)
    predicted_title = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f'True Title: {true_title}\n')
    print(f'Predicted Title: {predicted_title}\n')
    print(f'Abstract: {abstract}\n\n\n')