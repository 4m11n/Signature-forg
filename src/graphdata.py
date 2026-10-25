import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

N_NODES = 128   # points sampled per signature
K = 10          # neighbours per node (includes the node itself)


def _farthest_point_sampling(pts, n):
    """Deterministic: start at the leftmost point, repeatedly add the point farthest from the chosen set."""
    m = len(pts)
    if m <= n:
        return np.resize(np.arange(m), n)      # cycle if too few points
    chosen = np.empty(n, dtype=np.int64)
    chosen[0] = np.argmin(pts[:, 0])
    dist = np.full(m, np.inf)
    for i in range(1, n):
        d = ((pts - pts[chosen[i - 1]]) ** 2).sum(1)
        dist = np.minimum(dist, d)
        chosen[i] = np.argmax(dist)
    return chosen


def image_to_graph(img_u8, n_nodes=N_NODES, k=K):
    """img_u8: 2-D uint8 grayscale image (dark ink on light paper).
    Returns coords [n_nodes, 2] float32 and neighbour index [n_nodes, k] int64."""
    # Otsu binarisation: removes dependence on absolute brightness
    _, ink = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(ink)
    if len(xs) < 16:                            # fallback: darkest 5% of pixels
        t = np.percentile(img_u8, 5)
        ys, xs = np.nonzero(img_u8 <= t)
    pts = np.stack([xs, ys], 1).astype(np.float32)

    sel = _farthest_point_sampling(pts, n_nodes)
    pts_s = pts[sel]

    # centre on the ink bounding box and scale uniformly to [-1, 1] (aspect ratio kept)
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    scale = max((hi - lo).max() / 2, 1.0)
    coords = (pts_s - centre) / scale

    d = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return coords.astype(np.float32), idx.astype(np.int64)


class GraphPairDataset(Dataset):
    """Same pairs CSV as SignaturePairDataset, but returns graphs instead of images."""
    def __init__(self, csv_path, img_size=105):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def _graph(self, path):
        if path not in self.cache:
            img = Image.open(path).convert("L").resize((self.img_size, self.img_size))
            self.cache[path] = image_to_graph(np.array(img))
        return self.cache[path]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ca, ia = self._graph(row["img_a"])
        cb, ib = self._graph(row["img_b"])
        return (torch.from_numpy(ca), torch.from_numpy(ia),
                torch.from_numpy(cb), torch.from_numpy(ib), row["label"])
