import torch
from copy import deepcopy

def orthogonalize(self):
    """
    Orthogonalisiert den hierarchischen Tuckertensor 'self', wenn dieser noch nicht orthogonal ist.

    Ein hierarchischer Tuckertensor wird orthogonal genannt, wenn die Spaltenraumbasen eines jeden Knotens orthogonal
    sind.
    ______________________________________________________________________
    Output:
    None
    ______________________________________________________________________
    Beispiel:
    x = HTTensor.randn((3,4,5,6))
    x.orthogonalize()
    """

    if self.is_orthog:
        # self ist bereits orthogonal, entsprechend kann direkt der neue HTucker Tensor
        # returned werden
        return self

    # Lesbarkeit
    x = self

    # Dict fuer die R Matrizen der QR-Zerlegungen
    R = {}

    # Orthogonalisieren der Blattmatrizen
    for leaf in x.dtree.get_leaves():
        # torch.linalg.qr gibt ein Tupel (Q,R) zurueck
            x.U[leaf], R[leaf] = torch.linalg.qr(x.U[leaf], mode="reduced")

    # Orthogonalisieren der Transfertensoren
    # Iteriere den Dimensionsbaum dazu bottom-up
    for level in range(x.dtree.get_depth()-1, -1, -1):
        for node in x.dtree.get_nodes_of_lvl(level):
            if x.dtree.is_leaf(node):
                # Blaetter wurden bereits orthogonalisiert
                continue
            # Kinder von node
            l, r = x.dtree.get_children(node)
            # Multipliziere R[l] und R[r] in den Transfertensor B[node]
            x.B[node] = torch.tensordot(R[r], x.B[node], dims=([1], [1]))
            x.B[node] = torch.tensordot(R[l], x.B[node], dims=([1], [1]))
            if not x.dtree.is_root(node):
                # Der Transfertensor der Wurzel muss nicht mehr orthogonalisiert werden
                # Daher wird dieser Abschnitt nur dann durchgefuehrt, falls node ungleich der Wurzel ist
                # Berechne also die QR Zerlegung der Matrizierung des geupdateten Transfertensors
                x.B[node], R[node] = torch.linalg.qr(self.matricise(x.B[node], t=(0, 1)), mode="reduced")
                # Dematriziere den orthogonalisierten Transfertensor wieder zu 3D
                x.B[node] = self.dematricise(x.B[node],
                                                      shape=(R[l].shape[0], R[r].shape[0], x.B[node].shape[1]), t=(0, 1))
            # Gebe Speicher den Speicher der R-Matrizen der Kinder frei
            del R[l]
            del R[r]
    # Setze die Flag, dass self ein orthogonaler HTucker Tensor ist
    x.is_orthog = True
    return x
