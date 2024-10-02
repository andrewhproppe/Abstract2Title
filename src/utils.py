from pathlib import Path
import platform

install_path = Path(__file__)
top = install_path.parents[1].absolute()

paths = {
    "datasets": top.joinpath("src/data/datasets"),
    "databases": top.joinpath("src/data/databases"),
    "models": top.joinpath("src/models"),
    "notebooks": top.joinpath("notebooks"),
    "scripts": top.joinpath("scripts"),
}