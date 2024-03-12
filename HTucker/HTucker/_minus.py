from copy import deepcopy
import torch

def minus(self, y):
    """
    Berechnet die Differenz der beiden hierarchischen Tuckertensoren 'self' und 'y'. Voraussetzung hierfuer ist, dass
    deren Dimensionsbaeume uebereinstimmen.
    Der hierarchische Rang der Differenz ergibt sich als Summe der beiden hierarchischen Raenge von 'self' und 'y'. Es
    ist also ggf. ratsam eine anschliessende Rangkuerzung durchzufuehren.
    ______________________________________________________________________
    Parameter:
    - y HTucker.HTTensor: Der Subtrahend der Differenz.
    ______________________________________________________________________
    Output:
    (HTucker.HTTensor,): Die Differenz gegeben als hierarchischer Tuckertensor.
    ______________________________________________________________________
    Beispiel:
                      HTucker.HTTensor         <~~~>          torch.Tensor
    a)
       x = HTTensor.randn((3,4,5,6))              |           x = torch.randn(3,4,5,6)
       y= HTTensor.randn((3,4,5,6))               |           y = torch.randn(3,4,5,6)
       summe = x.minus(y)                        |           summe = x - y

    b) Die Verwendung des Operators "-" ist
       moeglich
       x = HTTensor.randn((10,8,4))              |           x = torch.randn(10,8,4)
       y= HTTensor.randn((10,8,4))               |           y = torch.randn(10,8,4)
       summe = x - y                            |           summe = x - y
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

    # Erzeuge Kopien
    x = deepcopy(self)
    y = deepcopy(y)

    # Aus Lesbarkeitsgruenden
    rx = x.get_rank()
    ry = y.get_rank()

    # Konkatenieren der Blattmatrizen
    for leaf in x.dtree.get_leaves():
        x.U[leaf] = torch.hstack((x.U[leaf], y.U[leaf]))

    # 3D konkatenieren der Transfertensoren
    for node in x.dtree.get_inner_nodes():
        # Kinder
        l, r = x.dtree.get_children(node)
        if x.dtree.is_root(node):
            B = torch.zeros(rx[l] + ry[l], rx[r] + ry[r], 1)
            B[:rx[l], :rx[r]] = x.B[node]
            B[rx[l]:, rx[r]:] = -1.0 * y.B[node]
        else:
            B = torch.zeros(rx[l] + ry[l], rx[r] + ry[r], rx[node] + ry[node])
            B[:rx[l], :rx[r], :rx[node]] = x.B[node]
            B[rx[l]:, rx[r]:, rx[node]:] = y.B[node]
        x.B[node] = B

    # Setze is_orthog Flag auf False
    x.is_orthog = False
    return x

