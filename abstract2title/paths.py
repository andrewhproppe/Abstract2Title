from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_PATH = DATA_DIR / "datasets"
DATASETS_PATH.mkdir(parents=True, exist_ok=True)
