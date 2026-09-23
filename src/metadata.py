import os
import re
import pandas as pd

def build_metadata(root_dir):
    records = []
    pattern = re.compile(r'(original|forgeries)_(\d+)_(\d+)\.png', re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            m = pattern.match(fname)
            if not m:
                continue
            kind, writer_id, sample_id = m.groups()
            label = "genuine" if kind.lower() == "original" else "forged"
            records.append({
                "writer_id": int(writer_id),
                "sample_id": int(sample_id),
                "label": label,
                "filepath": os.path.join(dirpath, fname),
            })

    df = pd.DataFrame(records)
    assert len(df) > 0, "No files matched — check the filename pattern against dataset."
    return df

if __name__ == "__main__":
    df = build_metadata(r"C:\Users\arfan\OneDrive\Desktop\forgery\data\Cedar")
    print(df.groupby(["writer_id", "label"]).size().unstack())
    print(f"Total writers: {df.writer_id.nunique()}, total samples: {len(df)}")
    df.to_csv("cedar_metadata.csv", index=False)