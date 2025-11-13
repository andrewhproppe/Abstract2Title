import random

from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
from transformers import T5ForConditionalGeneration, T5Tokenizer

abstract_title_pairs = fetch_papers_for_training(limit=1000, db_name='arxiv_papers_1000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)
eval_df = papers.sample(frac=0.9, random_state=42)

# Load the pre-trained T5 model and tokenizer
model_name = "t5-base"  # You can also use "t5-base" or "t5-small" for smaller versions
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained("../model_training/results_T5/checkpoint-96")

for _ in range(5):
    random_idx = random.randint(0, len(eval_df)-1)

    abstract_text = eval_df.iloc[random_idx]['input_text']
    abstract = f"summarize: {abstract_text}"

    true_title = eval_df.iloc[random_idx]['target_text']

    inputs = tokenizer(abstract, return_tensors="pt", max_length=512, truncation=True)

    # Generate title
    output_ids = model.generate(inputs["input_ids"].to(model.device), max_length=128, num_beams=4, early_stopping=True)
    predicted_title = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f'True Title: {true_title}\n')
    print(f'Predicted Title: {predicted_title}\n')
    print(f'Abstract: {abstract}\n\n\n')