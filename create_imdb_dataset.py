#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main(source, target):
    source = Path(source).expanduser()
    target = Path(target).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    CLASSES = ["neg", "pos"]

    def get_texts(path):
        texts, labels = [], []
        it = (
            (idx, fname)
            for idx, label in enumerate(CLASSES)
            for fname in (path / label).glob("*.txt")
        )
        for idx, fname in tqdm(it, desc=f"Loading {path}", ncols=80, position=0):
            texts.append(fname.read_text(encoding="utf-8").replace("<br /><br />", "\n"))
            labels.append(idx)
        return pd.DataFrame({"label": labels, "text": texts})

    train = get_texts(source / "train")
    test = get_texts(source / "test")

    # dev, test = train_test_split(test, test_size=.5, random_state=42)

    train.to_csv(target / "train.tsv", "\t", index=False)
    # dev.to_csv(target / "dev.tsv", "\t", index=False)
    test.to_csv(target / "test.tsv", "\t", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", "-s", type=str, default="./aclImdb", help="Source IMDB directory (path to aclImdb)")
    parser.add_argument("--target", "-t", type=str, default="./glue_data/IMDB", help="Target directory (ex: glue_data/IMDB)")
    args = parser.parse_args()
    main(**vars(args))