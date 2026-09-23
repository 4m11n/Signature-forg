import torch
from model import SiameseCNN

model = SiameseCNN(embedding_dim=128)
model.eval()
dummy_a = torch.randn(1, 1, 105, 105)
dummy_b = torch.randn(1, 1, 105, 105)
torch.onnx.export(model, (dummy_a, dummy_b), "siamese_model.onnx")