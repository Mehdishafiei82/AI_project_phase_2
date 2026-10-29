
import pandas as pd
from pathlib import Path



OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = OUTPUT_DIR / "raw_data.csv"

REQUIRED_COLUMNS = [
    "user_id",
    "item_id",
    "item_type",
    "rating"
]


sample_data = {
    "user_id": [
        1, 1, 1, 2, 2, 2, 3, 3, 3, 4,
        4, 4, 5, 5, 5, 6, 6, 6, 7, 7
    ],
    "item_id": [
        101, 102, 103, 101, 104, 105, 102, 103, 106, 101,
        105, 107, 102, 104, 106, 103, 105, 107, 101, 106
    ],
    "item_type": [
        "movie", "book", "movie", "book", "movie",
        "book", "movie", "book", "movie", "movie",
        "book", "movie", "book", "movie", "book",
        "movie", "book", "movie", "book", "movie"
    ],
    "rating": [
        5, 4, 3, 4, 5, 3, 4, 5, 2, 3,
        4, 5, 4, 2, 5, 5, 3, 4, 4, 5
    ]
}

if RAW_FILE.exists():
    df = pd.read_csv(RAW_FILE)
    print("Loading existing raw_data.csv")
else:
    df = pd.DataFrame(sample_data)
    df.to_csv(RAW_FILE, index=False)
    print("Sample data created.")

initial_rows = len(df)


missing_columns = [
    col for col in REQUIRED_COLUMNS if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df[REQUIRED_COLUMNS].copy()

for col in ["user_id", "item_id", "item_type"]:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)


df["item_type"] = df["item_type"].str.lower()


df["rating"] = pd.to_numeric(
    df["rating"], errors="coerce"
)


before_missing = len(df)

df = df.dropna(
    subset=["user_id", "item_id", "item_type", "rating"]
).copy()

removed_missing = before_missing - len(df)


before_rating = len(df)

df = df[df["rating"].between(1, 5)].copy()

removed_invalid_ratings = before_rating - len(df)


before_type = len(df)

df = df[df["item_type"].isin(["movie", "book"])].copy()

removed_invalid_types = before_type - len(df)


before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["user_id", "item_id"],
    keep="first"
)

removed_duplicates = before_duplicates - len(df)


if df.empty:
    raise ValueError(
        "No valid data remains after cleaning."
    )


df = df.sort_values(
    by=["user_id", "item_id"]
).reset_index(drop=True)


CLEAN_FILE = OUTPUT_DIR / "cleaned_data.csv"
df.to_csv(CLEAN_FILE, index=False)




if len(df) == 1:
    train_df = df.copy()
    test_df = df.iloc[0:0].copy()
else:
    train_df = df.sample(
        frac=0.8, random_state=42
    )

    test_df = df.drop(train_df.index)


train_df.to_csv(
    OUTPUT_DIR / "train.csv", index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv", index=False
)


train_pairs = set(
    map(tuple, train_df[["user_id", "item_id"]].values)
)

test_pairs = set(
    map(tuple, test_df[["user_id", "item_id"]].values)
)

assert train_pairs.isdisjoint(test_pairs), (
    "Error: user-item pairs overlap between train and test."
)

assert df["rating"].between(1, 5).all()
assert df["item_type"].isin(["movie", "book"]).all()


report = pd.DataFrame({
    "metric": [
        "initial_rows",
        "final_rows",
        "removed_missing_values",
        "removed_invalid_ratings",
        "removed_invalid_item_types",
        "removed_duplicate_pairs",
        "training_rows",
        "testing_rows"
    ],
    "value": [
        initial_rows,
        len(df),
        removed_missing,
        removed_invalid_ratings,
        removed_invalid_types,
        removed_duplicates,
        len(train_df),
        len(test_df)
    ]
})

report.to_csv(
    OUTPUT_DIR / "cleaning_report.csv",
    index=False
)


print("\n--- Data Preparation Results ---")
print(f"Initial rows: {initial_rows}")
print(f"Cleaned rows: {len(df)}")
print(f"Training rows: {len(train_df)}")
print(f"Testing rows: {len(test_df)}")
print(f"Movies: {(df['item_type'] == 'movie').sum()}")
print(f"Books: {(df['item_type'] == 'book').sum()}")

print("\nOutput files:")
print("- raw_data.csv")
print("- cleaned_data.csv")
print("- train.csv")
print("- test.csv")
print("- cleaning_report.csv")

print("\nData preparation completed successfully.")
print("Note: Sample data is fictional.")
