import torch
from .dimtree import dimtree


def randn(cls, shape: tuple, rank: dict = None, is_orthog: bool = False):
    """
    Erzeugt einen hierarchischen Tuckertensor mit normalverteilten Eintraegen in den Blattmatrizen und Transfertensoren.
    Die Dimensionen werden hierbei als kanonischer Dimensionsbaum angeordnet.
    ______________________________________________________________________
    Parameter:
    - shape (int,...): Bestimmt die Dimensionsgroessen des zu erzeugenden hierarchischen Tuckertensors.
    - rank dict: Enthaelt die gewuenschten hierarchischen Raenge fuer die nicht-Wurzel-Knoten des zu erzeugenden
            hierarchischen Tuckertensors. Dabei muss nicht fuer jeden Knoten ein hierarchischer Rang angegeben werden.
            Unter Umstaenden kann es auftreten, dass der gewuenschte hierarchische Rang nicht moeglich ist, sodass der
            tatsaechliche hierarchische Rang niedriger ausfaellt.
            Wird dieser Parameter nicht uebergeben, werden hierarchische Raenge zufaellig zwischen 1 und 7 gewaehlt.
    - is_orthog bool: Bestimmt, ob der zu erzeugenden hierarchische Tuckertensor orthogonal sein soll oder nicht.
    ______________________________________________________________________
    Beispiel:
    - HTTensor.randn(shape=(3,4,5,6))
    - HTTensor.randn(shape=(3,4,5,6), rank={(0,): 10, (1,): 5, (2,): 6, (3,): 7, (0,1): 6, (2,3): 3}
    - HTTensor.randn(shape=(3,4,5,6), rank={(0,): 10, (0,1): 10}
    - HTTensor.randn(shape=(3,4,5,6), is_orthog=True)
    - HTTensor.randn(shape=(3,4,5,6), rank={(2,): 5, (3,): 10, (2,3): 7}, is_orthog=True)
    """
    if rank is None:
        rank = {}

    # Erzeuge kanonischen Dimensionsbaum
    order = len(shape)
    dtree = dimtree.get_canonic_dimtree(order)

    # Erzeuge Blattmatrizen
    U = {}
    for leaf in dtree.get_leaves():
        dim_sz = shape[leaf[0]]
        if leaf in rank:
            k = rank[leaf]
        else:
            k = torch.randint(1, min(dim_sz+1, 8), (1,)).item()
        U[leaf] = torch.randn(dim_sz, k)
        if is_orthog:
            U[leaf] = orthogonalize(U[leaf])

    # Erzeuge Transfertensoren
    B = {}
    for level in range(dtree.get_depth() - 1, -1, -1):
        for node in dtree.get_nodes_of_lvl(level):
            if dtree.is_leaf(node):
                continue
            l, r = dtree.get_children(node)
            if dtree.is_leaf(l):
                rank_l = U[l].shape[1]
            else:
                rank_l = B[l].shape[2]
            if dtree.is_leaf(r):
                rank_r = U[r].shape[1]
            else:
                rank_r = B[r].shape[2]
            if dtree.is_root(node):
                k = 1
            elif node in rank:
                k = min(rank_l * rank_r, rank[node])
            else:
                k = min(rank_l * rank_r, torch.randint(1, 8, (1,)).item())
            B[node] = torch.randn(rank_l, rank_r, k)
            if is_orthog:
                B[node] = orthogonalize(cls.matricise(B[node], t=(0, 1)))
                B[node] = cls.dematricise(A=B[node], shape=(rank_l, rank_r, k), t=(0, 1))

    # Erzeuge HTucker Objekt
    return cls(U=U, B=B, dtree=dtree, is_orthog=is_orthog)


def orthogonalize(A: torch.Tensor):
    """
    Hinweis: Das ist eine interne Funktion der Funktion HTucker.randn.
    Berechne orthogonale Basis des Spaltenraums von A.
    """
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q
