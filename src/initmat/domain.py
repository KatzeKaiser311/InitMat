import numpy as np
from scipy import sparse


def _cell_column(items):
    result = np.empty((len(items), 1), dtype=object)
    for i, item in enumerate(items):
        result[i, 0] = item
    return result


def create_domain_folded(domain_size):
    NX = NZsub = NZ = domain_size
    NZsupra = 0
    width = float(domain_size)
    int_bound = domain_size * np.ones(domain_size + 1)
    numV = (NX + 1) * (NZ + 1)
    numT = NX * NZ
    numTsub = NX * NZsub
    numTsupra = 0
    numE = NX * (NZ + 1) + NZ * (NX + 1)
    deltaX = width / NX
    numCE = 4 * numT

    coordV = np.zeros((numV, 2))
    coordV[:, 0] = np.tile(np.arange(0, width + deltaX, deltaX), NZ + 1)
    coordV[:, 1] = np.kron(np.arange(NZsub + 1), int_bound / NZsub)

    V0T = np.zeros((numT, 4))
    V0T[:, 0] = np.repeat(np.arange(NZ), NX) * (NX + 1) + np.tile(np.arange(1, NX + 1), NZ)
    V0T[:, 1] = V0T[:, 0] + 1
    V0T[:, 2] = V0T[:, 1] + NX + 1
    V0T[:, 3] = V0T[:, 2] - 1

    V0E = np.zeros((numE, 2))
    V0E[:numT] = V0T[:, :2]
    V0E[numT - 1:numT + NX] = V0T[(NZ - 1) * NX - 1:, 3:1:-1]
    vertical_start = NX * (NZ + 1)
    V0E[vertical_start:numE - NZ, 0] = V0T[:, 0]
    V0E[vertical_start:numE - NZ, 1] = V0T[:, 3]
    V0E[numE - NZ:, 0] = V0T[NX - 1::NX, 1]
    V0E[numE - NZ:, 1] = V0T[NX - 1::NX, 2]

    E0T = np.zeros((numT, 4))
    E0T[:, 0] = np.arange(1, numT + 1)
    E0T[:, 1] = E0T[:, 0] + NX
    E0T[:, 2] = E0T[:, 0] + NX * (NZ + 1) + 1
    E0T[:, 3] = E0T[:, 0] + NX * (NZ + 1)
    E0T[-NX:, 1] = np.arange(1, NX + 1)
    E0T[NX - 1::NX, 2] = E0T[0::NX, 3]

    CE0T = np.zeros((numT, 4))
    CE0T[:, 0] = np.arange(1, numT + 1)
    CE0T[:, 1] = CE0T[:, 0] + numT
    CE0T[:, 3] = CE0T[:, 1] + numT
    CE0T[:, 2] = CE0T[:, 3] + numT

    V0CE = np.zeros((numCE, 2))
    V0CE[:numT] = V0T[:, :2]
    V0CE[numT:2 * numT, 0] = V0T[:, 3]
    V0CE[numT:2 * numT, 1] = V0T[:, 2]
    V0CE[2 * numT:3 * numT, 0] = V0T[:, 0]
    V0CE[2 * numT:3 * numT, 1] = V0T[:, 3]
    V0CE[3 * numT:4 * numT, 0] = V0T[:, 1]
    V0CE[3 * numT:4 * numT, 1] = V0T[:, 2]

    idE = np.zeros((numE, 1))
    idE[NZsub * NX:(NZsub + 1) * NX] = -1
    idE[:NX] = 1
    idE[numE - NZ:] = 2
    idE[numT:numT + NX] = 4
    idE[numT + NX:numT + NX * (NZsub + 1):NX] = 6
    idE0T = idE.reshape(-1)[E0T.astype(int) - 1]

    coordV0T = coordV[V0T.astype(int) - 1]
    BA = coordV0T[:, 1, 1] - coordV0T[:, 0, 1]
    DA = coordV0T[:, 3, 1] - coordV0T[:, 0, 1]
    CB = coordV0T[:, 2, 1] - coordV0T[:, 1, 1]
    ACBD = coordV0T[:, 0, 1] + coordV0T[:, 2, 1] - coordV0T[:, 1, 1] - coordV0T[:, 3, 1]

    NuLengthE = (coordV[V0E[:, 1].astype(int) - 1] - coordV[V0E[:, 0].astype(int) - 1]) @ np.array([[0, -1], [1, 0]])
    NuLengthE0T = np.zeros((numT, 4, 2))
    NuLengthE0T[:, 0] = NuLengthE[E0T[:, 0].astype(int) - 1]
    NuLengthE0T[:, 1] = -NuLengthE[E0T[:, 1].astype(int) - 1]
    NuLengthE0T[:, 2] = NuLengthE[E0T[:, 2].astype(int) - 1]
    NuLengthE0T[:, 3] = -NuLengthE[E0T[:, 3].astype(int) - 1]
    lengthE0T = np.sqrt(np.sum(NuLengthE0T**2, axis=2))

    markE0TE0T = [sparse.lil_matrix((numT, numT)) for _ in range(4)]
    markE0TE0T[0][np.arange(NX, numT), np.arange(numT - NX)] = 1
    markE0TE0T[0][np.arange(NX), np.arange(numT - NX, numT)] = 1
    markE0TE0T[1][np.arange(numT - NX), np.arange(NX, numT)] = 1
    markE0TE0T[1][np.arange(numT - NX, numT), np.arange(NX)] = 1
    markE0TE0T[2][np.arange(numT - 1), np.arange(1, numT)] = 1
    markE0TE0T[2][np.arange(NX - 1, numT - NX, NX), np.arange(NX, numT - NX + 1, NX)] = 0
    markE0TE0T[2][np.arange(NX - 1, numT, NX), np.arange(0, numT - 1, NX)] = 1
    markE0TE0T[3][np.arange(1, numT), np.arange(numT - 1)] = 1
    markE0TE0T[3][np.arange(NX, numT - NX + 1, NX), np.arange(NX - 1, numT - NX, NX)] = 0
    markE0TE0T[3][np.arange(0, numT - 1, NX), np.arange(NX - 1, numT, NX)] = 1
    markE0TE0T = [item.tocsr() for item in markE0TE0T]

    areaT = 0.5 * deltaX * (DA + coordV0T[:, 2, 1] - coordV0T[:, 1, 1])
    baryE = 0.5 * (coordV[V0E[:, 0].astype(int) - 1] + coordV[V0E[:, 1].astype(int) - 1])
    baryE0T = np.zeros((numT, 4, 2))
    for i in range(4):
        baryE0T[:, i] = baryE[E0T[:, i].astype(int) - 1]
    baryT = ((coordV0T[:, 0] + coordV0T[:, 1] + coordV0T[:, 2] + coordV0T[:, 3]) / 4)[:, None, :]

    markE0Tint1D = np.ones((NX, 2))
    markE0Tint1D[0, 1] = 0
    markE0Tint1D[-1, 0] = 0
    mark1D1 = sparse.lil_matrix((NX, NX))
    mark1D2 = sparse.lil_matrix((NX, NX))
    mark1D1[np.arange(NX - 1), np.arange(1, NX)] = 1
    mark1D2[np.arange(1, NX), np.arange(NX - 1)] = 1
    coordL1D = np.arange(NX) * deltaX

    T0E = np.zeros((numE, 2))
    T0E[:numT, 0] = np.arange(1, numT + 1)
    T0E[NX:numT + NX, 1] = np.arange(1, numT + 1)
    T0E[:NX, 1] = np.arange(numT - NX + 1, numT + 1)
    T0E[vertical_start:numE - NZ, 0] = np.arange(1, numT + 1)
    T0E[vertical_start:numE - NZ, 1] = np.arange(numT)
    T0E[vertical_start:numE - NZ:NX, 1] = np.arange(1, numT, NX)
    T0E[numE - NZ:, 1] = np.arange(NX, NZ * NX + 1, NX)
    T0E[T0E[:, 0] == 0, 0] = T0E[T0E[:, 0] == 0, 1]
    T0E[T0E[:, 1] == 0, 1] = T0E[T0E[:, 1] == 0, 0]

    return {
        "NX": float(NX), "NZsupra": float(NZsupra), "numV": float(numV), "numT": float(numT), "numTsub": float(numTsub), "numTsupra": float(numTsupra), "numE": float(numE), "deltaX": float(deltaX), "numCE": float(numCE),
        "coordV": coordV, "V0T": V0T, "V0E": V0E, "E0T": E0T, "CE0T": CE0T, "V0CE": V0CE, "idE": idE, "idE0T": idE0T,
        "coordV0T": coordV0T, "coordV0Tsub": coordV0T, "coordV0Tsupra": coordV0T[0:0],
        "BA": BA[:, None], "DA": DA[:, None], "ACBD": ACBD[:, None], "CB": CB[:, None], "BAsub": BA[:, None], "BAsupra": BA[0:0, None], "DAsub": DA[:, None], "DAsupra": DA[0:0, None], "ACBDsub": ACBD[:, None], "ACBDsupra": ACBD[0:0, None], "CBsupra": CB[0:0, None],
        "NuLengthE0T": NuLengthE0T, "NuLengthE0Tsub": NuLengthE0T, "NuLengthE0Tsupra": NuLengthE0T[0:0], "lengthE0T": lengthE0T, "lengthE0Tsub": lengthE0T, "lengthE0Tsupra": lengthE0T[0:0],
        "markE0TE0T": _cell_column(markE0TE0T), "markE0TE0Tsub": _cell_column(markE0TE0T), "markE0TE0Tsupra": _cell_column([item[0:0, 0:0] for item in markE0TE0T]),
        "areaT": areaT[:, None], "areaTsub": areaT[:, None], "areaTsupra": areaT[0:0, None], "baryE0T": baryE0T, "baryE0Tsub": baryE0T, "baryE0Tsupra": baryE0T[0:0], "baryT": baryT, "baryTsub": baryT,
        "markE0Tint1D": markE0Tint1D, "markE0TE0T1D": _cell_column([mark1D1.tocsr(), mark1D2.tocsr()]), "coordL1D": coordL1D[None, :], "T0E": T0E,
    }