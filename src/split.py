import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("cedar_metadata.csv")

gss = GroupShuffleSplit(n_splits=1, test_size=10/55, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["writer_id"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# writer leakage check
overlap = set(train_df.writer_id) & set(test_df.writer_id)
assert len(overlap) == 0, f"Writer leakage detected: {overlap}"

print(f"Train writers: {train_df.writer_id.nunique()}, Test writers: {test_df.writer_id.nunique()}")
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
train_df.to_csv("train_metadata.csv", index=False)
test_df.to_csv("test_metadata.csv", index=False)
print("Split OK — no writer overlap.")