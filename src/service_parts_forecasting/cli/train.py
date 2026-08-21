from __future__ import annotations

import argparse

from .common import add_experiment_arguments, run_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and roll-evaluate a global forecast model")
    add_experiment_arguments(parser, default_stage="validation")
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()

