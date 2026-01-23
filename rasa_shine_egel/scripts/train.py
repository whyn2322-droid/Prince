import pathlib
import sys
import importlib._bootstrap as _bootstrap

from rasa.api import train


def main() -> None:
    # Work around environment issues where importlib._bootstrap.sys is altered.
    _bootstrap.sys = sys
    project_root = pathlib.Path(__file__).resolve().parents[1]
    train(
        domain=str(project_root / "domain.yml"),
        config=str(project_root / "config.yml"),
        training_files=str(project_root / "data"),
        output=str(project_root / "models"),
    )


if __name__ == "__main__":
    main()
