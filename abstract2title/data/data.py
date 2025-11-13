import json
import random
import re
import sqlite3

import pandas as pd
from tqdm import tqdm

from abstract2title.paths import DATA_DIR
from abstract2title.src.utils import paths


def load_arxiv_metadata(filename, n_papers, random_select=False, categories_filter=None):
    """
    Load arXiv metadata from a JSON file, optionally filter by category, and
    return a dataset containing the title, abstract, and journal title for a given number of papers.

    Parameters:
    - filename (str): Path to the JSON file containing arXiv metadata.
    - n_papers (int): The number of papers to extract.
    - random_select (bool): If True, select papers randomly. Otherwise, select them in order.
    - categories_filter (str): If provided, filter to include only papers with this category (e.g., 'quant-ph').

    Returns:
    - dataset (list of dicts): A list where each entry is a dictionary with 'title', 'abstract', and 'journal-ref'.
    """
    dataset = []

    with open(filename, "r") as file:
        print("Loading arXiv metadata...")
        lines = file.readlines()
        print("Done loading metadata.")

        # Filter by category if a filter is specified
        if categories_filter:
            print(f"Filtering for category: {categories_filter}")
            filtered_lines = [
                line
                for line in lines
                if categories_filter in json.loads(line).get("categories", "")
            ]
            print(f'Found {len(filtered_lines)} entries with category "{categories_filter}".')
        else:
            filtered_lines = lines

        # Select lines either randomly or sequentially
        if random_select:
            selected_lines = random.sample(filtered_lines, min(n_papers, len(filtered_lines)))
        else:
            selected_lines = filtered_lines[:n_papers]

        # Extract the relevant information for each paper
        for line in tqdm(selected_lines, desc="Extracting papers"):
            paper = json.loads(line)
            entry = {
                "title": paper.get("title", "No title available"),
                "abstract": paper.get("abstract", "No abstract available"),
                "journal_ref": paper.get("journal-ref", "No journal reference available"),
                "categories": paper.get("categories", "No categories available"),
            }
            dataset.append(entry)

    return dataset


def save_dataset_to_file(dataset, filename_base, type="csv"):
    """
    Converts a list of dictionaries into a Pandas DataFrame and saves it as both a CSV and Parquet file.

    Parameters:
    - dataset (list of dicts): The list of dictionaries to convert and save.
    - filename_base (str): The base filename (without extension) for saving the dataset.

    Returns:
    - None
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(dataset)

    # Save as CSV
    if type == "csv":
        csv_filename = f"{filename_base}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Dataset saved as CSV: {csv_filename}")

    elif type == "parquet":
        parquet_filename = f"{filename_base}.parquet"
        df.to_parquet(parquet_filename, index=False)
        print(f"Dataset saved as Parquet: {parquet_filename}")


def load_dataset(filename, n_rows=None, random_select=False):
    """
    Loads a dataset from a CSV or Parquet file and optionally selects a subset of rows.

    Parameters:
    - filename (str): Path to the file (CSV or Parquet).
    - n_rows (int, optional): The number of rows to load. If None, loads the entire dataset.
    - random_select (bool): If True, selects rows randomly. Otherwise, selects them in order.

    Returns:
    - pd.DataFrame: The loaded (and optionally subsetted) dataset.
    """

    root_dir = paths.get("datasets")
    filename_full = root_dir.joinpath(filename)

    # Load data based on file extension
    if filename.endswith(".csv"):
        df = pd.read_csv(filename_full)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(filename_full)
    else:
        raise ValueError("Unsupported file format. Please use a .csv or .parquet file.")

    # If n_rows is specified, select a subset of rows
    if n_rows is not None:
        if random_select:
            # Randomly sample rows without replacement
            df = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
        else:
            # Select the first n_rows rows
            df = df.head(n_rows)

    return df


def load_arxiv_metadata_into_db(
    filename, n_papers, random_select=False, db_name="arxiv_papers.db", root_dir="datasets"
):
    """
    Load arXiv metadata from a JSON file and insert directly into an SQLite database.

    Parameters:
    - filename (str): Path to the JSON file containing arXiv metadata.
    - n_papers (int): The number of papers to extract.
    - random_select (bool): If True, select papers randomly. Otherwise, select them in order.
    - db_name (str): Name of the SQLite database.
    - root_dir (str): Directory where the SQLite database is stored.
    """

    conn = sqlite3.connect(paths.get("datasets").joinpath(db_name))
    c = conn.cursor()

    # Create the table if it doesn't exist
    c.execute("""CREATE TABLE IF NOT EXISTS papers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  abstract TEXT,
                  journal_ref TEXT,
                  categories TEXT)""")

    with open(filename, "r") as file:
        print("Loading arXiv metadata...")
        lines = file.readlines()
        print("Done.")

        if random_select:
            selected_lines = random.sample(lines, n_papers)
        else:
            selected_lines = lines[:n_papers]

        # Insert papers directly into the database
        for line in selected_lines:
            paper = json.loads(line)
            title = paper.get("title", "No title available")
            abstract = paper.get("abstract", "No abstract available")
            journal_ref = paper.get("journal-ref", "No journal reference available")
            categories = paper.get("categories", "No categories available")

            c.execute(
                """INSERT INTO papers(title, abstract, journal_ref, categories)
                         VALUES(?, ?, ?, ?)""",
                (title, abstract, journal_ref, categories),
            )

    conn.commit()
    conn.close()
    print(f"{n_papers} papers successfully inserted into the database.")


def print_paper_info(dataset, index):
    """
    Print the title, abstract, and journal reference of a paper at a given index in the dataset.

    Parameters:
    - dataset (list of dicts): The dataset containing the papers' information.
    - index (int): The index of the paper to print.
    """
    paper = dataset[index]
    print(f"Paper {index + 1}")
    print(f"Title: {paper['title']}")
    print(f"Abstract: {paper['abstract']}")
    print(f"Journal Reference: {paper['journal_ref']}")
    print(f"Categories: {paper['categories']}")
    print("-" * 80)


def create_database(db_name="arxiv_papers.db", root_dir="datasets"):
    conn = sqlite3.connect(paths.get("datasets").joinpath(db_name))
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS papers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  abstract TEXT,
                  journal_ref TEXT,
                  categories TEXT)""")
    conn.commit()
    conn.close()


def insert_paper(conn, paper):
    sql = """INSERT INTO papers(title, abstract, journal_ref, categories)
             VALUES(?, ?, ?, ?)"""
    cur = conn.cursor()
    cur.execute(sql, (paper["title"], paper["abstract"], paper["journal_ref"], paper["categories"]))
    conn.commit()


def save_dataset_to_db(dataset, db_name="arxiv_papers.db", root_dir="datasets"):
    conn = sqlite3.connect(paths.get("datasets").joinpath(db_name))

    # print('Saving dataset to database...')
    for paper in tqdm(dataset, desc="Inserting papers"):
        insert_paper(conn, paper)

    conn.close()
    print("Done.")


def fetch_papers(db_name="arxiv_papers.db", limit=10, root_dir="datasets"):
    """
    Fetch a limited number of papers from the database.

    Parameters:
    - db_name (str): The name of the SQLite database file.
    - limit (int): The number of papers to fetch.

    Returns:
    - results (list of tuples): A list of fetched papers.
    """
    conn = sqlite3.connect(paths.get("datasets").joinpath(db_name))
    c = conn.cursor()

    c.execute("SELECT * FROM papers LIMIT ?", (limit,))
    results = c.fetchall()

    conn.close()
    return results


def print_paper_info_db(paper_tuple):
    """
    Print the title, abstract, and journal reference of a paper fetched from the database.

    Parameters:
    - paper_tuple (tuple): A tuple containing the paper's id, title, abstract, and journal_ref.
    """
    paper_id, title, abstract, journal_ref = paper_tuple
    print(f"Paper ID: {paper_id}")
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print(f"Journal Reference: {journal_ref}")
    print("-" * 80)


def fetch_papers_for_training(db_name="arxiv_papers.db", limit=100):
    """
    Fetch the abstract and title of a limited number of papers from the database.

    Parameters:
    - db_name (str): The name of the SQLite database file.
    - limit (int): The number of papers to fetch.
    - root_dir (str): The directory where the SQLite database is stored.

    Returns:
    - results (list of tuples): A list of fetched papers (abstract, title).
    """
    db_path = paths.get("datasets").joinpath(db_name)
    print(f"Loading {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT abstract, title FROM papers LIMIT ?", (limit,))
    results = c.fetchall()

    conn.close()
    return results


def preprocess_text(text):
    # Remove extra spaces, newlines, and special characters
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
    text = text.strip()  # Remove leading/trailing spaces
    return text


def abstract_title_pairs_to_pandas(abstract_title_pairs):
    abstracts, titles = zip(*abstract_title_pairs)
    papers = pd.DataFrame({"abstract": abstracts, "title": titles})
    papers = papers[["abstract", "title"]]
    papers.columns = ["input_text", "target_text"]
    papers = papers.dropna()
    return papers
