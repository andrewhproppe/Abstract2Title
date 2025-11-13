import argparse

from abstract2title.data.data import (
    create_database,
    load_arxiv_metadata,
    print_paper_info,
    save_dataset_to_db,
    save_dataset_to_file,
)
from abstract2title.paths import DATA_DIR, DATASETS_PATH

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-papers",
        type=int,
        help="Number of papers to extract",
    )
    parser.add_argument(
        "--cat-filter",
        type=str,
        required=False,
        help="Optional category filter for retrieved papers.",
    )
    parser.add_argument(
        "--random-select",
        action="store_true",
        help="Randomize paper ordering.",
    )
    parser.add_argument(
        "--print-result",
        type=int,
        required=False,
        help="Optionally print of the papers for verification.",
    )
    args = parser.parse_args()

    # Specify the file and number of papers to extract
    arxiv_file = DATA_DIR / "raw/259/arxiv-metadata-oai-snapshot.json"
    db_name = DATASETS_PATH / f"arxiv_papers_{args.num_papers}n_{args.cat_filter}_cat"

    # Load the metadata
    dataset = load_arxiv_metadata(
        arxiv_file,
        args.num_papers,
        args.random_select,
        categories_filter=args.cat_filter,
    )

    # Create the database and save the dataset
    create_database(db_name=db_name)
    save_dataset_to_db(dataset, db_name=db_name)

    # # Save as csv
    # save_dataset_to_file(dataset, db_name, type="csv")

    # Print details for all papers from dataset
    if args.print_result is not None:
        for i in range(args.print_result):
            print_paper_info(dataset, i)
