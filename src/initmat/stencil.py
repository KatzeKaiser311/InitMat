import numpy as np


def stencil(nx, nzd, candidates, layers=1):
    candidates = np.asarray(candidates, dtype=int).reshape(-1)
    offsets = [(0, 0)]

    for layer in range(1, layers + 1):
        for dr in range(-layer, layer + 1):
            dc = layer - abs(dr)
            offsets.append((dr, 0)) if dc == 0 else offsets.extend(((dr, -dc), (dr, dc)))

    rows, cols = candidates % nx, candidates // nx
    dr = np.array([x[0] for x in offsets], dtype=int)
    dc = np.array([x[1] for x in offsets], dtype=int)

    return ((cols[:, None] + dc) % nzd) * nx + (rows[:, None] + dr) % nx