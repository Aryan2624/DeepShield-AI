from pathlib import Path

import pandas as pd


DATASET_DIR = Path(r"E:\DeepShield-Datasets\phishing_messages")


csv_files = sorted(DATASET_DIR.glob("*.csv"))

print("=" * 80)
print("DEEPSHIELD AI - PHISHING DATASET INSPECTION")
print("=" * 80)

print(f"\nDataset folder: {DATASET_DIR}")
print(f"CSV files found: {len(csv_files)}\n")


for file_path in csv_files:
    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    try:
        df = pd.read_csv(
            file_path,
            nrows=5,
            low_memory=False,
        )

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nNumber of columns:")
        print(len(df.columns))

        print("\nSample:")
        print(df.head())

        print("\nData types:")
        print(df.dtypes)

    except Exception as error:
        print(f"\nCould not read file:")
        print(error)

    print("\n")