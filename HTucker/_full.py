import torch

def full(self):
    """
    Gibt den vollen Tensor zurueck.
    :return: torch.Tensor
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