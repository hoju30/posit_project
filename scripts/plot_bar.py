import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dot_error_bar.csv", help="pred CSV (target,pred)")
    parser.add_argument("--out-prefix", default="dot_error_bar", help="output file prefix")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.csv))

    
    plt.figure()
    plt.bar(df["target"], df["pred"])
    plt.yscale("log")
    plt.xlabel("target")
    plt.ylabel("relative error (log scale)")
    plt.title("Predicted relative errors (log)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_log.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
