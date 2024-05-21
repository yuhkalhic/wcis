import os
import os.path as op
import sys
import tarfile
import time
import urllib
import numpy as np
import pandas as pd
from tqdm import tqdm


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return

    duration = time.time() - start_time
    progress_size = int(count * block_size)
    if duration > 0:
        speed = progress_size / (1024.0**2 * duration)
        percent = int(count * block_size * 100.0 / total_size)
        sys.stdout.write(
            f"\r{percent}% | {progress_size / (1024.**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
    else:
        # 如果时间为0，避免除零错误
        sys.stdout.write(
            f"\r... | {progress_size / (1024.**2):.2f} MB | Calculating speed... | {duration:.2f} sec elapsed"
        )
    sys.stdout.flush()


def download_dataset():
    source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target = "aclImdb_v1.tar.gz"

    if os.path.exists(target):
        os.remove(target)

    if not os.path.isdir("aclImdb") and not os.path.isfile("aclImdb_v1.tar.gz"):
        urllib.request.urlretrieve(source, target, reporthook)

    if not os.path.isdir("aclImdb"):
        with tarfile.open(target, "r:gz") as tar:
            tar.extractall()


def load_dataset_into_dataframe():
    basepath = "aclImdb"
    labels = {"pos": 1, "neg": 0}
    df = pd.DataFrame()

    with tqdm(total=50000) as pbar:
        for s in ("test", "train"):
            for l in ("pos", "neg"):
                path = os.path.join(basepath, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                        txt = infile.read()
                    df = pd.concat([df, pd.DataFrame([[txt, labels[l]]], columns=["review", "sentiment"])], ignore_index=True)
                    pbar.update()

    df.columns = ["text", "label"]
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))

    print("Class distribution:")
    print(np.bincount(df["label"].values))

    return df


def partition_dataset(df):
    df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df_train = df_shuffled.iloc[:35000]
    df_val = df_shuffled.iloc[35000:40000]
    df_test = df_shuffled.iloc[40000:]

    if not op.exists("data"):
        os.makedirs("data")
    df_train.to_csv(op.join("data", "train.csv"), index=False, encoding="utf-8")
    df_val.to_csv(op.join("data", "val.csv"), index=False, encoding="utf-8")
    df_test.to_csv(op.join("data", "test.csv"), index=False, encoding="utf-8")


def get_dataset():
    files = ("test.csv", "train.csv", "val.csv")
    download = not all(op.exists(f) for f in files)

    if download:
        download_dataset()
        df = load_dataset_into_dataframe()
        partition_dataset(df)

    df_train = pd.read_csv(op.join("data", "train.csv"))
    df_val = pd.read_csv(op.join("data", "val.csv"))
    df_test = pd.read_csv(op.join("data", "test.csv"))

    return df_train, df_val, df_test

