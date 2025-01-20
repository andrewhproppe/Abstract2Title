from pathlib import Path
import platform

install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "raw": top.joinpath("src/data/raw"),
    "datasets": top.joinpath("src/data/datasets"),
    "models": top.joinpath("src/models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts"),
}