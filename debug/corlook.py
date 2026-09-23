import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import pandas as pd

df = pd.read_csv("metadata/cedar_metadata.csv")
path = df[df.label == "genuine"].iloc[0]["filepath"]
print(f"Using: {path}")

# PIL pipeline (matches dataset.py / training)
pil_transform = T.Compose([T.Grayscale(1), T.Resize((105,105)), T.ToTensor()])
pil_tensor = pil_transform(Image.open(path))

# cv2 pipeline (matches corruption_eval.py, no augmentation applied)
cv_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv_img = cv2.resize(cv_img, (105, 105))
cv_tensor = torch.from_numpy(cv_img).float().unsqueeze(0) / 255.0

print("PIL tensor stats:", pil_tensor.mean().item(), pil_tensor.std().item(), pil_tensor.min().item(), pil_tensor.max().item())
print("CV2 tensor stats:", cv_tensor.mean().item(), cv_tensor.std().item(), cv_tensor.min().item(), cv_tensor.max().item())
print("Max abs difference:", (pil_tensor - cv_tensor).abs().max().item())