import platform
from pathlib import Path

install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("data/raw"),
    "datasets": top.joinpath("data/datasets"),
    "models": top.joinpath("src/models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts"),
}
