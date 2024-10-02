import json
import random
import sqlite3
import re

from tqdm import tqdm
from src.utils import paths

def load_arxiv_metadata(filename, n_papers, random_select=False):
    """
    Load arXiv metadata from a JSON file and return a dataset containing
    the title, abstract, and journal title for a given number of papers.

    Parameters:
    - filename (str): Path to the JSON file containing arXiv metadata.
    - n_papers (int): The number of papers to extract.
    - random_select (bool): If True, select papers randomly. Otherwise, select them in order.

    Returns:
    - dataset (list of dicts): A list where each entry is a dictionary with 'title', 'abstract', and 'journal-ref'.
    """
    dataset = []

    with open(filename, 'r') as file:
        print('Loading arXiv metadata...')
        lines = file.readlines()
        print('Done.')

        if random_select:
            selected_lines = random.sample(lines, n_papers)
        else:
            selected_lines = lines[:n_papers]

        for line in tqdm(selected_lines, desc='Extracting papers'):
            paper = json.loads(line)
            entry = {
                'title': paper.get('title', 'No title available'),
                'abstract': paper.get('abstract', 'No abstract available'),
                'journal_ref': paper.get('journal-ref', 'No journal reference available')
            }
            dataset.append(entry)

    return dataset


def load_arxiv_metadata_into_db(filename, n_papers, random_select=False, db_name='arxiv_papers.db', root_dir='databases'):
    """
    Load arXiv metadata from a JSON file and insert directly into an SQLite database.

    Parameters:
    - filename (str): Path to the JSON file containing arXiv metadata.
    - n_papers (int): The number of papers to extract.
    - random_select (bool): If True, select papers randomly. Otherwise, select them in order.
    - db_name (str): Name of the SQLite database.
    - root_dir (str): Directory where the SQLite database is stored.
    """
    
    conn = sqlite3.connect(paths.get("databases").joinpath(db_name))
    c = conn.cursor()

    # Create the table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS papers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  abstract TEXT,
                  journal_ref TEXT)''')

    with open(filename, 'r') as file:
        print('Loading arXiv metadata...')
        lines = file.readlines()
        print('Done.')

        if random_select:
            selected_lines = random.sample(lines, n_papers)
        else:
            selected_lines = lines[:n_papers]

        # Insert papers directly into the database
        for line in selected_lines:
            paper = json.loads(line)
            title = paper.get('title', 'No title available')
            abstract = paper.get('abstract', 'No abstract available')
            journal_ref = paper.get('journal-ref', 'No journal reference available')

            c.execute('''INSERT INTO papers(title, abstract, journal_ref)
                         VALUES(?, ?, ?)''', (title, abstract, journal_ref))

    conn.commit()
    conn.close()
    print(f'{n_papers} papers successfully inserted into the database.')


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
    print("-" * 80)


def create_database(db_name='arxiv_papers.db', root_dir='databases'):
    conn = sqlite3.connect(paths.get("databases").joinpath(db_name))
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS papers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  abstract TEXT,
                  journal_ref TEXT)''')

    conn.commit()
    conn.close()


def insert_paper(conn, paper):
    sql = '''INSERT INTO papers(title, abstract, journal_ref)
             VALUES(?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (paper['title'], paper['abstract'], paper['journal_ref']))
    conn.commit()


def save_dataset_to_db(dataset, db_name='arxiv_papers.db', root_dir='databases'):
    conn = sqlite3.connect(paths.get("databases").joinpath(db_name))

    for paper in dataset:
        insert_paper(conn, paper)

    conn.close()


def fetch_papers(db_name='arxiv_papers.db', limit=10, root_dir='databases'):
    """
    Fetch a limited number of papers from the database.

    Parameters:
    - db_name (str): The name of the SQLite database file.
    - limit (int): The number of papers to fetch.

    Returns:
    - results (list of tuples): A list of fetched papers.
    """
    conn = sqlite3.connect(paths.get("databases").joinpath(db_name))
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


def fetch_papers_for_training(db_name='arxiv_papers.db', limit=100, root_dir='databases'):
    """
    Fetch the abstract and title of a limited number of papers from the database.

    Parameters:
    - db_name (str): The name of the SQLite database file.
    - limit (int): The number of papers to fetch.
    - root_dir (str): The directory where the SQLite database is stored.

    Returns:
    - results (list of tuples): A list of fetched papers (abstract, title).
    """
    conn = sqlite3.connect(paths.get("databases").joinpath(db_name))
    c = conn.cursor()

    c.execute("SELECT abstract, title FROM papers LIMIT ?", (limit,))
    results = c.fetchall()

    conn.close()
    return results


def preprocess_text(text):
    # Remove extra spaces, newlines, and special characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = text.strip()  # Remove leading/trailing spaces
    return text