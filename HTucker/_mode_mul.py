import torch
from copy import deepcopy


def mode_mul(self, A, dim):
    """
    Berechnet die Modusmultiplikation des durch self repraesentierten hierarchischen Tuckertensors mit der Matrix A
    entlang des Modus dim. A o_dim self.
    Hinweis: Die Operationen werden auf einer Kopie von self durchgefuehrt.
    :param self: HTucker.HTucker
    :param A: 2D-torch.Tensor
    :param dim: int
    :return: HTucker.HTucker
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("Argument 'A': type(A)={} | A ist kein torch.Tensor.".format(type(A)))
    if not isinstance(dim, int):
        raise TypeError("Argument 'dim': type(dim)={} | dim ist kein int.".format(type(dim)))
    if len(A.shape) != 2:
        raise ValueError("Argument 'A': A.shape={} | A ist kein 2D-torch.Tensor.".format(A.shape))
    if dim not in range(self.get_order()):
        raise ValueError("Argument 'dim': dim={} | dim ist keine gueltige Dimension fuer einen HTucker Tensor"
                         "der Ordnung {}.".format(dim, self.get_order()))
    if self.get_shape()[dim] != A.shape[1]:
        raise ValueError("Argument 'A', 'dim': A.shape={}, dim={}, self.shape={} | A, dim und self"
                         " passen nicht zusammen.".format(A.shape, dim, self.get_shape()))
    # Kopiere Blattmatrixdict, Transfertensordict und dtree von self
    U, B, dtree = deepcopy(self.U), deepcopy(self.B), deepcopy(self.dtree)

    # Multipliziere A mit der Blattmatrix des Knotens, der die Dimension dim repraesentiert
    node = (dim,)
    U[node] = torch.tensordot(A, U[node], dims=([1], [0]))
    return type(self)(U=U, B=B, dtree=dtree, is_orthog=False)
