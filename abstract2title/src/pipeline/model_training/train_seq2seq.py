import argparse
import datetime
import random

import pandas as pd
import torch
from simpletransformers.seq2seq import Seq2SeqModel

from abstract2title.data.data import fetch_papers_for_training
from abstract2title.paths import DATA_DIR


def abstract_title_pairs_to_pandas(abstract_title_pairs):
    abstracts, titles = zip(*abstract_title_pairs)
    papers = pd.DataFrame({"abstract": abstracts, "title": titles})
    papers = papers[["abstract", "title"]]
    papers.columns = ["input_text", "target_text"]
    papers = papers.dropna()
    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-name",
        type=str,
        required=True,
        help="Name of the training dataset.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Optional name for model tagging.",
    )
    args = parser.parse_args()

    abstract_title_pairs = fetch_papers_for_training(limit=100, db_name=args.db_name)

    papers = abstract_title_pairs_to_pandas(abstract_title_pairs)

    eval_df = papers.sample(frac=0.1, random_state=42)
    train_df = papers.drop(eval_df.index)

    if args.model_name is None:
        name = datetime.datetime.now().strftime("%Y-%m-%d")
    else:
        name = args.model_name

    model_dir = DATA_DIR / "model_checkpoints" / name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "save_model_every_epoch": True,
        "save_eval_checkpoints": False,
        "max_seq_length": 512,
        "train_batch_size": 2,
        "num_train_epochs": args.num_epochs,
        "output_dir": str(model_dir),
    }

    # Create a Bart-base model
    print("\n Starting model training")
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-base",
        args=model_args,
        use_cuda=True if torch.cuda.is_available() else False,
    )

    # Train the model
    model.train_model(
        train_df,
        save_model_every_epoch=True,
    )

    # Evaluate the model
    result = model.eval_model(eval_df[:5])
    print(result)

    for _ in range(5):
        random_idx = random.randint(0, len(eval_df) - 1)

        abstract = eval_df.iloc[random_idx]["input_text"]
        true_title = eval_df.iloc[random_idx]["target_text"]

        # Predict with trained BART model
        predicted_title = model.predict([abstract])[0]

        print(f"True Title: {true_title}\n")
        print(f"Predicted Title: {predicted_title}\n")
        print(f"Abstract: {abstract}\n\n\n")


if __name__ == "__main__":
    main()
