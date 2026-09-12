import time
from pathlib import Path

import pandas as pd

from features import extract_url_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "malicious_phish_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "phishing_urls"
    / "url_features_v2.csv"
)

CHUNK_SIZE = 50_000


def preprocess_dataset():
    print("=" * 75)
    print("DeepShield AI - URL Feature Engineering v2")
    print("=" * 75)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found:\n{INPUT_PATH}"
        )

    if OUTPUT_PATH.exists():
        print("\nExisting v2 feature file found.")
        print("Removing old file before regeneration...")

        OUTPUT_PATH.unlink()

    total_rows = 0
    chunk_number = 0

    start_time = time.time()

    reader = pd.read_csv(
        INPUT_PATH,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        chunk_number += 1

        chunk_start = time.time()

        if "url" not in chunk.columns:
            raise ValueError(
                "Input dataset does not contain 'url' column."
            )

        if "type" not in chunk.columns:
            raise ValueError(
                "Input dataset does not contain 'type' column."
            )

        urls = chunk["url"].astype(str)

        feature_records = [
            extract_url_features(url)
            for url in urls
        ]

        feature_df = pd.DataFrame.from_records(
            feature_records
        )

        if feature_df.shape[1] != 34:
            raise ValueError(
                f"Expected 34 features, "
                f"but extracted {feature_df.shape[1]}."
            )

        feature_df.insert(
            0,
            "url",
            chunk["url"].values,
        )

        feature_df["type"] = (
            chunk["type"].values
        )

        feature_df.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=not OUTPUT_PATH.exists(),
            index=False,
        )

        rows_processed = len(chunk)

        total_rows += rows_processed

        chunk_time = (
            time.time() - chunk_start
        )

        print(
            f"Chunk {chunk_number:02d} | "
            f"Rows: {rows_processed:,} | "
            f"Total: {total_rows:,} | "
            f"Time: {chunk_time:.2f} sec"
        )

    total_time = time.time() - start_time

    print("\n" + "=" * 75)
    print("FEATURE ENGINEERING V2 COMPLETE")
    print("=" * 75)

    print(
        f"Total rows processed : "
        f"{total_rows:,}"
    )

    print(
        "ML features          : 34"
    )

    print(
        f"Total time           : "
        f"{total_time:.2f} sec"
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    preprocess_dataset()