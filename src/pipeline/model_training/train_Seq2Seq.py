import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from src.data.data import fetch_papers_for_training
from simpletransformers.seq2seq import Seq2SeqModel

def abstract_title_pairs_to_pandas(abstract_title_pairs):
    abstracts, titles = zip(*abstract_title_pairs)
    papers = pd.DataFrame({
        'abstract': abstracts,
        'title': titles
    })
    papers = papers[['abstract', 'title']]
    papers.columns = ['input_text', 'target_text']
    papers = papers.dropna()
    return papers

abstract_title_pairs = fetch_papers_for_training(limit=100, db_name='arxiv_papers_50000n.db')

papers = abstract_title_pairs_to_pandas(abstract_title_pairs)

eval_df = papers.sample(frac=0.1, random_state=42)
train_df = papers.drop(eval_df.index)


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "save_model_every_epoch": True,
    "save_eval_checkpoints": False,
    "max_seq_length": 512,
    "train_batch_size": 6,
    "num_train_epochs": 3,
}

# Create a Bart-base model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",
    args=model_args,
)

# Train the model
model.train_model(
    train_df,
    save_model_every_epoch=True,
)

# Evaluate the model
result = model.eval_model(eval_df[:5])
print(result)


import random

for _ in range(5):

    random_idx = random.randint(0, len(eval_df)-1)

    abstract = eval_df.iloc[random_idx]['input_text']
    true_title = eval_df.iloc[random_idx]['target_text']

    # Predict with trained BART model
    predicted_title = model.predict([abstract])[0]

    print(f'True Title: {true_title}\n')
    print(f'Predicted Title: {predicted_title}\n')
    print(f'Abstract: {abstract}\n\n\n')