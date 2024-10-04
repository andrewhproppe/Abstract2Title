from src.data.data import load_arxiv_metadata, print_paper_info, create_database, save_dataset_to_db


# Example usage:
if __name__ == "__main__":
    # Specify the file and number of papers to extract
    arxiv_file = 'datasets/arxiv-metadata-oai-snapshot.json'
    n_papers = 50000
    random_select = True  # Set to False if you want papers in order
    db_name = f'arxiv_papers_{n_papers}n.db'

    # Load the metadata
    dataset = load_arxiv_metadata(arxiv_file, n_papers, random_select)

    # Print details for all papers from dataset
    # for i in range(n_papers):
    #     print_paper_info(dataset, i)

    # Create the database and save the dataset
    create_database(db_name=db_name)
    save_dataset_to_db(dataset, db_name=db_name)

    # # Fetch and print papers from the database
    # papers_from_db = fetch_papers(limit=5, db_name=db_name)
    # for paper in papers_from_db:
    #     print_paper_info_db(paper)