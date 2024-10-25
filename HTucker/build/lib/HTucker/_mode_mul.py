import torch
from copy import deepcopy


def mode_mul(self, A: torch.Tensor, dim: int):
    """
    Berechnet die Modusmultiplikation von 'self' mit der Matrix 'A' entlang der Dimension 'dim': A o_dim self
    Voraussetzung dafuer ist, dass 'A' so viele Spalten hat, wie die Dimension 'dim' von 'self' gross ist.
    ______________________________________________________________________
    Parameter:
    - A 2D torch.Tensor: Ein 2D torch.Tensor mit A.shape[1] == self.get_shape()[dim]
    - dim int: Ein integer mit 0 < dim < self.get_order()-1
    ______________________________________________________________________
    Output:
    (HTucker.HTTensor,): Das berechnete Modusprodukt.
    ______________________________________________________________________
    Beispiel:
              HTucker.HTTensor                   <~~~>                   torch.Tensor
    a) x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
       A = torch.randn((7,5))                    |           A = torch.randn(7,5)
       prod = x.mode_mul(A,dim=2)                |           prod = torch.tensordot(x, A, dims=[[2],[1]])
       prod.shape    # = (3,4,7,6)               |           prod.shape    # = torch.size([3,4,6,7])
                                                 |           prod = torch.transpose(prod, dim0=3, dim1=2)
                                                 |           prod.shape    # = torch.size([3,4,7,6])

    a) x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
       A = torch.randn((10,3))                   |           A = torch.randn(10,3)
       prod = x.mode_mul(A,dim=0)                |           prod = torch.tensordot(x, A, dims=[[0],[1]])
       prod.shape    # = (10,4,7,6)              |           prod.shape    # = torch.size([4,6,7,10])
                                                 |           prod = torch.transpose(prod, dim0=3, dim1=0)
                                                 |           prod.shape    # = torch.size([10,4,7,6])
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

    # Fuer bessere Lesbarkeit
    x = self

    # Multipliziere A mit der Blattmatrix des Knotens, der die Dimension dim repraesentiert
    node = (dim,)
    x.U[node] = torch.tensordot(A, x.U[node], dims=([1], [0]))
    # Setze is_orthog Flag auf False
    x.is_orthog = False
    return x
