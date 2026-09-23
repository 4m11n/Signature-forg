import pandas as pd
import itertools
import random

random.seed(42)

def build_pairs(df, pairs_per_writer=30):
    pairs = []
    writers = df["writer_id"].unique()

    for w in writers:
        genuine = df[(df.writer_id == w) & (df.label == "genuine")]["filepath"].tolist()
        forged = df[(df.writer_id == w) & (df.label == "forged")]["filepath"].tolist()

        genuine_combos = list(itertools.combinations(genuine, 2))
        random.shuffle(genuine_combos)
        for a, b in genuine_combos[:pairs_per_writer // 2]:
            pairs.append({"img_a": a, "img_b": b, "label": 1, "writer_id": w})

        gf_combos = list(itertools.product(genuine, forged))
        random.shuffle(gf_combos)
        for a, b in gf_combos[:pairs_per_writer // 2]:
            pairs.append({"img_a": a, "img_b": b, "label": 0, "writer_id": w})

    pairs_df = pd.DataFrame(pairs)
    return pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)

if __name__ == "__main__":
    train_df = pd.read_csv("metadata/train_metadata.csv")
    test_df = pd.read_csv("metadata/test_metadata.csv")

    train_pairs = build_pairs(train_df, pairs_per_writer=40)
    test_pairs = build_pairs(test_df, pairs_per_writer=40)

    print("Train pairs label balance:\n", train_pairs.label.value_counts())
    print("Test pairs label balance:\n", test_pairs.label.value_counts())

    train_pairs.to_csv("metadata/train_pairs.csv", index=False)
    test_pairs.to_csv("metadata/test_pairs.csv", index=False)