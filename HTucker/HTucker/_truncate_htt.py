import torch
from copy import deepcopy
from math import sqrt


def truncate_htt(self, opts: dict):
    """
    Fuehrt eine Rangkuerzung auf dem hierarchischen Tuckertensor 'self' durch. Die dabei einzuhaltenden Constraints
    finden sich im Parameter 'opts'.
    ______________________________________________________________________
    Parameter:
    - opts dict: Enthaelt mindestens eine der folgenden Optionen:
                                    - "max_rank": positiver integer | Legt den maximalen hierarchischen Rang
                                                  fest
                                    - "err_tol_abs": positiver float | Legt die einzuhaltende absolute
                                                     Fehlertoleranz fest
                                    - "err_tol_rel": positiver float | Left die einzuhaltende relative
                                                     Fehlertoleranz fest
    ______________________________________________________________________
    Output:
    None
    ______________________________________________________________________
    Beispiel:
    x = torch.randn(10,10,10,10)
    xh = HTTensor.truncate(x)
    xh.get_rank()    # = {(0, 1, 2, 3): 1, (0,): 10, (1,): 10, (2,): 10, (3,): 10, (0, 1): 100, (2, 3): 100}
    opts = {"max_rank": 25, "err_tol_abs": 10.0}
    xh.truncate_htt(opts)
    xh.get_rank()    # = {(0, 1, 2, 3): 1, (0,): 10, (1,): 10, (2,): 10, (3,): 10, (0, 1): 25, (2, 3): 25}
    """

    # Anpassen der Fehlertoleranzen in opts
    # Soll global der Fehler e eingehalten werden, muss der Kuerzungsfehler pro Knoten
    # kleiner gleich e / sqrt((Tensorordnung * 2 - 2)) bleiben
    opts = {k: (v/sqrt(self.get_order()*2-3) if k in ["err_tol_abs", "err_tol_rel"]
                else v) for k, v in opts.items()}

    # Fuer bessere Lesbarkeit
    x = self

    # Orthogonalisiere self, falls notwendig
    if not x.is_orthog:
        x.orthogonalize()

    # Berechne die reduzierten Gram'schen Matrizen
    G = x._get_gramians()

    # Iteriere durch den Dimensionsbaum bottom up
    for level in range(x.dtree.get_depth(), 0, -1):
        for node in x.dtree.get_nodes_of_lvl(level):
            # Berechne linke Singulaervektoren
            Q, sv = x.left_svd_gramian(G[node])
            rank = x._get_truncation_rank(sv, opts)
            Q = Q[:, :rank]
            if x.dtree.is_leaf(node):
                # Kuerze Blattmatrix durch Multiplikation mit Q
                x.U[node] = x.U[node] @ Q
            else:
                x.B[node] = torch.tensordot(x.B[node], Q, dims=([2], [0]))
            # Update Transfertensor des Elternknotens
            par = x.dtree.get_parent(node)
            if x.dtree.is_left(node):
                x.B[par] = torch.tensordot(Q.T, x.B[par], dims=([1], [0]))
            else:
                x.B[par] = torch.tensordot(Q.T, x.B[par], dims=([1], [1]))
                x.B[par] = torch.movedim(x.B[par], source=0, destination=1)

    # Setze is_orthog Flag auf False
    x.is_orthog = False
