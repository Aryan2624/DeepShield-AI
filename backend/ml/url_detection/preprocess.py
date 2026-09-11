from pathlib import Path
from time import perf_counter

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
    / "url_features.csv"
)

CHUNK_SIZE = 50_000


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw URLs into numerical ML features.
    """

    feature_rows = [
        extract_url_features(url)
        for url in chunk["url"]
    ]

    feature_df = pd.DataFrame(feature_rows)

    # Keep original URL for later error analysis.
    feature_df.insert(
        0,
        "url",
        chunk["url"].values,
    )

    # Target label goes at the end.
    feature_df["type"] = chunk["type"].values

    return feature_df


def preprocess_dataset():
    print("=" * 65)
    print("DeepShield AI - URL Feature Preprocessing")
    print("=" * 65)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found:\n{INPUT_PATH}"
        )

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    total_rows = 0
    chunk_number = 0

    start_time = perf_counter()

    reader = pd.read_csv(
        INPUT_PATH,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        chunk_number += 1

        print(
            f"\nProcessing chunk {chunk_number} "
            f"({len(chunk):,} URLs)..."
        )

        processed_chunk = process_chunk(
            chunk
        )

        processed_chunk.to_csv(
            OUTPUT_PATH,
            mode="a",
            index=False,
            header=(chunk_number == 1),
        )

        total_rows += len(
            processed_chunk
        )

        elapsed = (
            perf_counter()
            - start_time
        )

        print(
            f"Completed rows: "
            f"{total_rows:,}"
        )

        print(
            f"Elapsed time: "
            f"{elapsed:.2f} seconds"
        )


    total_time = (
        perf_counter()
        - start_time
    )

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETE")
    print("=" * 65)

    print(
        f"Total rows processed : "
        f"{total_rows:,}"
    )

    print(
        f"Total processing time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"\nFeature dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    preprocess_dataset()