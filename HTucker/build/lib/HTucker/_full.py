import torch

def full(self):
    """
    Berechnet den durch 'self' repraesentierten vollen Tensor.
    ______________________________________________________________________
    Output:
    (torch.Tensor,): Der volle Tensor.
    ______________________________________________________________________
    Beispiel:
    a)
    x = HTTensor.randn((3,4,5,6))
    x_full = x.full()
    type(x_full)    # = torch.Tensor
    b)
    x = HTTensor.truncate(torch.randn(5,6,7,8))
    x_full = x.full()
    type(x_full)    # = torch.Tensor
    """

    # Wiederherstellung der Basen der Matrizierungen
    U = {}
    for level in range(self.dtree.get_depth(), -1, -1):
        for t in self.dtree.get_nodes_of_lvl(level):
            if self.dtree.is_leaf(t):
                U[t] = self.U[t]
            else:
                left, right = self.dtree.get_children(t)
                UrB = torch.tensordot(U[right], self.B[t], dims=([1], [1]))
                UlUrB = torch.tensordot(U[left], UrB, dims=([1], [1]))
                U[t] = self.matricise(UlUrB, (0, 1))
                del U[right]
                del U[left]
    # Reshapen der Basis der Wurzel-Matrizierung in Originalform
    x = self.dematricise(U[self.dtree.get_root()], self.get_shape(), self.dtree.get_root())
    return x