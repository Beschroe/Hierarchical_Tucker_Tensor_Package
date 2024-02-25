from math import prod
import torch
from copy import deepcopy


def scalar_mul(self, c):
    """
    Multipliziert den hierarchischen Tuckertensor self mit dem Skalar c.
    Hinweis: Die Multiplikation wird auf einer Kopie von self durchgefuehrt.
    :param self: HTucker.HTucker
    :param c: float, int, torch.Tensor mit einem Eintrag
    :param copy: bool
    :return: HTucker.HTucker
    """
    if isinstance(c, torch.Tensor):
        if prod(c.shape) != 1:
            raise ValueError("Argument 'c': c.shape={} | Es koennen nur torch.Tensor Objekte mit genau einem Eintrag "
                             "skalarmultipliziert werden.".format(c.shape))
    if type(c) not in [int, float, torch.Tensor]:
        raise TypeError("Argument 'c': type(c)={} | c ist weder int, float noch ein torch.Tensor.".format(type(c)))

    # Kopiere Blattmatrixdict und Transfertensordict sowie dtree von self
    U, B, dtree = deepcopy(self.U), deepcopy(self.B), deepcopy(self.dtree)
    # Multiplizierre den Transfertensor der Wurzel mit dem Skalar
    B[dtree.get_root()] = B[dtree.get_root()] * c

    return type(self)(U=U, B=B, dtree=dtree)
