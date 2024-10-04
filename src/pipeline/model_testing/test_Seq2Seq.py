import random

from src.data.data import fetch_papers_for_training, abstract_title_pairs_to_pandas
from simpletransformers.seq2seq import Seq2SeqModel

abstract_title_pairs = fetch_papers_for_training(limit=50000, db_name='arxiv_papers_50000n.db')
papers = abstract_title_pairs_to_pandas(abstract_title_pairs)
eval_df = papers.sample(frac=0.9, random_state=42)

# Create a Bart-base model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="../model_training/outputs",
)

for _ in range(5):
    random_idx = random.randint(0, len(eval_df)-1)

    abstract = eval_df.iloc[random_idx]['input_text']
    true_title = eval_df.iloc[random_idx]['target_text']

    # Predict with trained BART model
    predicted_title = model.predict([abstract])[0]

    print(f'True Title: {true_title}\n')
    print(f'Predicted Title: {predicted_title}\n')
    print(f'Abstract: {abstract}\n\n\n')