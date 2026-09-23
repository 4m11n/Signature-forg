import torch
from dataset import SignaturePairDataset
from model import SiameseCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseCNN(embedding_dim=128).to(device)
model.load_state_dict(torch.load("siamese_model.pth"))
model.eval()

ds = SignaturePairDataset("metadata/test_pairs.csv")

same_idx = ds.df[ds.df.label == 1].index[:5].tolist()
diff_idx = ds.df[ds.df.label == 0].index[:5].tolist()

with torch.no_grad():
    print("--- SAME (label=1) pairs ---")
    for i in same_idx:
        img_a, img_b, label = ds[i]
        emb_a, emb_b = model(img_a.unsqueeze(0).to(device), img_b.unsqueeze(0).to(device))
        dist = torch.nn.functional.pairwise_distance(emb_a, emb_b)
        print(f"Pair {i} | dist={dist.item():.4f}")

    print("--- DIFFERENT (label=0) pairs ---")
    for i in diff_idx:
        img_a, img_b, label = ds[i]
        emb_a, emb_b = model(img_a.unsqueeze(0).to(device), img_b.unsqueeze(0).to(device))
        dist = torch.nn.functional.pairwise_distance(emb_a, emb_b)
        print(f"Pair {i} | dist={dist.item():.4f}")