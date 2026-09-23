import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SignaturePairDataset
from model import SiameseCNN

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, label):
        # label: 1 = same/genuine pair, 0 = different/forged pair (matches your pairs CSV convention)
        dist = F.pairwise_distance(emb_a, emb_b)
        same_loss = label * dist.pow(2)
        diff_loss = (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        loss = 0.5 * (same_loss + diff_loss)
        return loss.mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SignaturePairDataset("metadata/train_pairs.csv")
    test_ds = SignaturePairDataset("metadata/test_pairs.csv")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = SiameseCNN(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for img_a, img_b, label in train_loader:
            img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device).float()

            optimizer.zero_grad()
            emb_a, emb_b = model(img_a, img_b)
            loss = criterion(emb_a, emb_b, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "siamese_model.pth")
    print("Model saved to siamese_model.pth")

if __name__ == "__main__":
    train()