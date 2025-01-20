from src.data.data import load_arxiv_metadata, save_dataset_to_file

# Example usage:
if __name__ == "__main__":
    # Specify the file and number of papers to extract
    arxiv_file = 'raw/204/arxiv-metadata-oai-snapshot.json'
    n_papers = 100000
    random_select = True  # Set to False if you want papers in order
    categories_filter = 'quant-ph'  # Set to None if you want all categories
    # categories_filter = None
    db_name = f'databases/arxiv_papers_{n_papers}n_{categories_filter}_cat'

    # Load the metadata
    dataset = load_arxiv_metadata(arxiv_file, n_papers, random_select, categories_filter=categories_filter)

    # Save as csv
    save_dataset_to_file(dataset, db_name, type='csv')

    # Print details for all papers from dataset
    # for i in range(n_papers):
    #     print_paper_info(dataset, i)

    # # Create the database and save the dataset
    # create_database(db_name=db_name)
    # save_dataset_to_db(dataset, db_name=db_name)

    # # Fetch and print papers from the database
    # papers_from_db = fetch_papers(limit=5, db_name=db_name)
    # for paper in papers_from_db:
    #     print_paper_info_db(paper)
