# Abstract-to-title

Use different LLM architectures (BART, T5, GPT2) to predict the title of a paper from its abstract. Trained using the open-source arXiv dataset from KaggleHub.

There are training scripts for BART, T5, and GPT2 models. The BART seq2seq model performed best, and much of the post-training content of this repo has only been developed and tested for that model.

## Installation

In a virtual environment, install packages and dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get raw data and create a database

Download the arXiv dataset:
```
# Get the raw data
python data/raw/download_arxiv_data.py 

# Create a database - here are some default parameters
python data/make_database.py --num-papers 1000 --cat-filter quant-ph --random-select --print-result 10

> [!NOTE]
> You may need to update the path the arXiv json file here

```

## Train seq2seq model

```
python src/pipeline/model_training/train_seq2seq.py --db-name arxiv_papers_1000n_quant-ph_cat --num-epochs 5 --name mymodel
```

## Run the app to make some predictions!

Make sure `app.py` is pointing to the correct model path, and from the main directory, run:
```
uvicorn app:app --reload
```