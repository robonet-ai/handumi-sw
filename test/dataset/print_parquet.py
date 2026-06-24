import sys
import numpy as np
import pandas as pd

DEFAULT_PATH = "/home/leonardo/NONHUMAN/dexumi/datasets/dexumi_demo/meta/episodes/chunk-000/file-000.parquet"


def describe_value(val):
    if isinstance(val, np.ndarray):
        return f"np.ndarray shape={val.shape} dtype={val.dtype}"
    if isinstance(val, (list, tuple)):
        arr = np.array(val)
        return f"{type(val).__name__} -> np.array shape={arr.shape} dtype={arr.dtype}"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{type(val).__name__} = {val}"
    return repr(val)


def print_parquet(path: str):
    df = pd.read_parquet(path)

    print(f"File : {path}")
    print(f"Rows : {len(df)}")
    print(f"Cols : {list(df.columns)}")
    print("-" * 60)

    for col in df.columns:
        sample = df[col].iloc[0]
        print(f"\n[{col}]")
        print(f"  dtype    : {df[col].dtype}")
        print(f"  sample   : {describe_value(sample)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print_parquet(path)
