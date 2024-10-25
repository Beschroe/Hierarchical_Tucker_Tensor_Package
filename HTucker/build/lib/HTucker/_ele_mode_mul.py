import torch
from copy import deepcopy

def ele_mode_mul(self, v: torch.Tensor, dim: int):
    """
    Berechnet das elementweise Produkt des hierarchischen Tuckertensors 'self' mit dem 1D-torch.Tensor 'v' entlang der
    Dimension 'dim'. Voraussetzung dafuer ist, dass 'v' so viele Eintraege hat, wie die Dimension 'dim' von 'self' gross
    ist.
    ______________________________________________________________________
    Parameter:
    - v torch.Tensor: Ein 1D torch.Tensor mit v.shape[0] == self.get_shape()[dim]
    - dim int: Ein integer mit 0 < dim < self.get_order()-1
    ______________________________________________________________________
    Output:
    (HTucker.HTTensor,): Das berechnete elementweise Produkt.
    ______________________________________________________________________
    Beispiel:
              HTucker.HTTensor                   <~~~>                   torch.Tensor
    a) x = HTTensor.randn((3,4,5,6))              |           x = torch.randn(3,4,5,6)
       v = torch.randn((5,))                      |           v = torch.randn(5)
       prod = x.ele_mode_mul(v=v,dim=2)           |           prod = x * v[None,None,:,None]

    b) x = HTTensor.randn((10,8,4))               |           x = torch.randn(10,8,4)
       v = torch.randn((10,))                     |           v = torch.randn(10)
       prod = x.ele_mode_mul(v=v,dim=0)           |           prod = x * v[:,None,None]
    """

    if not isinstance(v, torch.Tensor):
        raise TypeError("Argument 'A': type(A)={} | A ist kein torch.Tensor.".format(type(v)))
    if not isinstance(dim, int):
        raise TypeError("Argument 'dim': type(dim)={} | dim ist kein int.".format(type(dim)))
    if len(v.shape) != 1:
        raise ValueError("Argument 'A': A.shape={} | A ist kein 1D-torch.Tensor.".format(v.shape))
    if dim not in range(self.get_order()):
        raise ValueError("Argument 'dim': dim={} | dim ist keine gueltige Dimension fuer einen HTucker Tensor"
                         "der Ordnung {}.".format(dim, self.get_order()))
    if self.get_shape()[dim] != v.shape[0]:
        raise ValueError("Argument 'A', 'dim': A.shape={}, dim={}, self.shape={} | A, dim und self"
                         " passen nicht zusammen.".format(v.shape, dim, self.get_shape()))

    # Aus Gruenden der Lesbarkeit
    x = deepcopy(self)

    # Multipliziere A elementweise mit der Blattmatrix des Knotens, der die Dimension dim repraesentiert
    node = (dim,)
    x.U[node] = x.U[node] * v[:, None]
    # Setze is_orthog Flag auf False
    x.is_orthog = False
    return x
