import torch
import warnings


def _get_gramians(self):
    """
    Hinweis: Dies ist eine interne Funktion
    ______________________________________________________________________
    Berechnet die reduzierten Gram'schen Matrizen des hierarchischen Tuckertensors 'self'. Dabei wird 'self' en passant
    orthogonalisiert, sofern 'self' noch kein orthogonaler hierarchischer Tuckertensor ist.

    Sei X ein Tensor und X_t die zugehoerige t-Matrizierung. Ferner sei U_t eine zugehoerige orthogonale Basis
    des Spaltenraums von X_t. Dann erfuellt die reduzierte Gram'sche Matrix G_t folgende Gleichung:
    X_t @ X_t.T = U_t @ G_t @ U_t.T
    ______________________________________________________________________
    Output:
    (dict,): Das dict enthaelt fuer jeden Knoten des Dimensionsbaums von 'self' die zugehoerige
             reduzierte Gram'sche Matrix.
    ______________________________________________________________________
    Beispiel:
    X = torch.randn(3,4,5,6)
    Xh = HTTensor.truncate(X)
    G = Xh._get_gramians()
    t = (0,)
    X_t = HTTensor.matricise(X,t)
    G_t = G[t]
    U_t = Xh.U[t]
    torch.allclose(X_t @ X_t.T, U_t @ G_t @ U_t.T)    # is True
    """
    x = self

    # Orthogonalisiere x, falls notwendig
    if not x.is_orthog:
        x.orthogonalize()

    # Gramian dict
    G = {x.dtree.get_root(): torch.ones(1, 1)}

    # Traversiere den Dimensionsbaum top down beginnend bei der Wurzel
    # Berechne dabei die jeweiligen reduzierten Gram'schen Matrizen
    for level in range(0, x.dtree.get_depth(), 1):
        for node in x.dtree.get_nodes_of_lvl(level):
            if x.dtree.is_leaf(node):
                # Gram'sche Matrix wurde bereits beim Elternknoten berechnet
                continue
            # Kinder von Node
            l, r = x.dtree.get_children(node)
            # Kontrahiere den Transfertensor von node mit der reduzierten Gramschen'Matrix von node
            BG = torch.tensordot(x.B[node], G[node], dims=([2], [1]))
            # Be
            G[l] = torch.tensordot(x.B[node], BG, dims=([1, 2], [1, 2]))
            G[r] = torch.tensordot(x.B[node], BG, dims=([0, 2], [0, 2]))
    return G
