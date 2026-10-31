#!/usr/bin/env python
import argparse
from pathlib import Path
from urllib.request import urlretrieve

URLS = {
    "men.txt": "https://raw.githubusercontent.com/SimonPreissner/fruitfly/master/data/MEN_dataset_natural_form_full",
    "wordsim353.csv": "https://raw.githubusercontent.com/msavva/transphoner/master/data/wordsim353.csv",
    "simlex999.txt": "https://raw.githubusercontent.com/rishibommasani/Contextual2Static/master/SimLex-999.txt",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/benchmarks")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = out / name
        print(f"Downloading {name}...")
        urlretrieve(url, path)
        print(path)
    print("STS-B is loaded directly from sentence-transformers/stsb by evaluate_similarity.py")


if __name__ == "__main__":
    main()
