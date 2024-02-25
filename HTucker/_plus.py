import torch
from copy import deepcopy


def plus(self, y):
    """
    Addiere die beiden hierarchischen Tuckertensoren self und y. Damit dies moeglich ist, muessen die beiden Dimensions-
    baeume uebereinstimmen.
    :param self: HTucker.HTucker
    :param y: HTucker.HTucker
    :return: HTucker.HTucker
    """
    # Typecheck
    if not isinstance(y, type(self)):
        raise TypeError("Argument 'y': type(y)={} | y ist nicht vom Typ HTucker.HTucker.".format(type(y)))
    # Kompatibilitaetscheck
    if not self.dtree.is_equal(y.dtree):
        raise ValueError("Argument 'y': Der Dimensionsbaum von y ist nicht kompatibel.")
    if self.get_shape() != y.get_shape():
        raise ValueError("Argument 'y': y.shape={} | Die shape von y ist nicht kompatibel zur shape von"
                         "self={}.".format(y.get_shape(), self.get_shape()))

    # Aus Lesbarkeitsgruenden
    x = self
    rx = x.get_rank()
    ry = y.get_rank()

    # Blattmatrixdict, Transfertensordict und Dimtree des resultierenden HTucker Tensors
    U, B, dtree = {}, {}, deepcopy(x.dtree)

    # Konkatenieren der Blattmatrizen
    for leaf in dtree.get_leaves():
        U[leaf] = torch.hstack((x.U[leaf], y.U[leaf]))

    # 3D konkatenieren der Transfertensoren
    for node in dtree.get_inner_nodes():
        # Kinder
        l, r = dtree.get_children(node)
        if dtree.is_root(node):
            B[node] = torch.zeros(rx[l]+ry[l], rx[r]+ry[r], 1)
            B[node][:rx[l], :rx[r]] = x.B[node]
            B[node][rx[l]:, rx[r]:] = y.B[node]
        else:
            B[node] = torch.zeros(rx[l]+ry[l], rx[r]+ry[r], rx[node]+ry[node])
            B[node][:rx[l], :rx[r], :rx[node]] = x.B[node]
            B[node][rx[l]:, rx[r]:, rx[node]:] = y.B[node]

    # Konstruieren des summierten HTucker Tensors
    z = type(self)(U=U, B=B, dtree=dtree, is_orthog=False)
    return z