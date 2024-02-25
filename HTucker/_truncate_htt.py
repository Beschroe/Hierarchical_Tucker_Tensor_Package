import torch
from math import sqrt


def truncate_htt(self, opts):
    """
    Kuerzt den hierarchischen Tuckertensor self auf einen niedrigeren hierarchischen Rang. Das Argument opts enthaelt
    die hierbei einzuhaltenden Constraints.
    opts enthaelt folgende Optionen:
                                    - "max_rank": positiver integer | Legt den maximalen hierarchischen Rang
                                                  fest
                                    - "err_tol_abs": positiver float | Legt die einzuhaltende absolute
                                                     Fehlertoleranz fest
                                    - "err_tol_rel": positiver float | Left die einzuhaltende relative
                                                     Fehlertoleranz fest
    :param self: HTucker.HTucker
    :param opts: dict
    :return: HTucker.HTucker
    """
    # Anpassen der Fehlertoleranzen in opts
    # Soll global der Fehler e eingehalten werden, muss der Kuerzungsfehler pro Knoten
    # kleiner gleich e / sqrt((Tensorordnung * 2 - 2)) bleiben
    opts = {k: (v/sqrt(self.get_order()*2-2) if k in ["err_tol_abs", "err_tol_rel"]
                else v) for k, v in opts.items()}

    # Orthogonalisiere self, falls notwendig
    if not self.is_orthog:
        self.orthogonalize()

    # Fuer bessere Lesbarkeit
    U = self.U
    B = self.B
    dtree = self.dtree

    # Berechne die reduzierten Gram'schen Matrizen
    G = self.get_gramians()

    # Iteriere durch den Dimensionsbaum bottom up
    for level in range(dtree.get_depth(), 0, -1):
        for node in dtree.get_nodes_of_lvl(level):
            # Berechne linke Singulaervektoren
            Q, sv = self.left_svd_gramian(G[node])
            rank = self.get_truncation_rank(sv, opts)
            Q = Q[:, :rank]
            if dtree.is_leaf(node):
                # Kuerze Blattmatrix durch Multiplikation mit Q
                U[node] = U[node] @ Q
            else:
                B[node] = torch.tensordot(B[node], Q, dims=([2], [0]))
            # Update Transfertensor des Elternknotens
            par = dtree.get_parent(node)
            if dtree.is_left(node):
                B[par] = torch.tensordot(Q.T, B[par], dims=([1], [0]))
            else:
                B[par] = torch.tensordot(Q.T, B[par], dims=([1], [1]))
                B[par] = torch.movedim(B[par], source=0, destination=1)

    # Gebe gekuerzten HTucker Tensor zurueck
    return self