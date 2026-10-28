"""Download experiment datasets once before parallel workers start."""

import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10, MNIST

from utils.paths import repo_root


DATASET_ROOT = repo_root(Path(__file__)) / "data" / "datasets"


def download_dataset(name: str) -> None:
    normalized = name.replace(".", "-").lower()
    if normalized == "cifar-10":
        root = DATASET_ROOT / "CIFAR"
        CIFAR10(root, train=True, download=True)
        CIFAR10(root, train=False, download=True)
    elif normalized == "mnist":
        MNIST(DATASET_ROOT, train=True, download=True)
        MNIST(DATASET_ROOT, train=False, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="dataset names used by the experiment presets")
    args = parser.parse_args()

    # Preserve the requested order while avoiding duplicate downloads.
    for name in dict.fromkeys(args.datasets):
        print(f"Preparing dataset: {name}", flush=True)
        download_dataset(name)


if __name__ == "__main__":
    main()
