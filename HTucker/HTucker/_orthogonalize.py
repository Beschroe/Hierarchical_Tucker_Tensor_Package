import torch


def orthogonalize(self):
    """
    Orthogonalisiere self, falls self ein nicht-orthogonaler hierarchischer Tuckertensor ist.
    :param self: HTucker.HTucker
    :return: HTucker.HTucker (orthogonal)
    """

    if self.is_orthog:
        # self ist bereits orthogonal, entsprechend kann direkt der neue HTucker Tensor
        # returned werden
        return self

    # Dict fuer die R Matrizen der QR-Zerlegungen
    R = {}

    # Fuer die Lesbarkeit...
    U, B, dtree = self.U, self.B, self.dtree
    # Orthogonalisieren der Blattmatrizen
    for leaf in dtree.get_leaves():
        # torch.linalg.qr gibt ein Tupel (Q,R) zurueck
            U[leaf], R[leaf] = torch.linalg.qr(U[leaf], mode="reduced")

    # Orthogonalisieren der Transfertensoren
    # Iteriere den Dimensionsbaum dazu bottom-up
    for level in range(dtree.get_depth()-1, -1, -1):
        for node in dtree.get_nodes_of_lvl(level):
            if dtree.is_leaf(node):
                # Blaetter wurden bereits orthogonalisiert
                continue
            # Kinder von node
            l, r = dtree.get_children(node)
            # Multipliziere R[l] und R[r] in den Transfertensor B[node]
            B[node] = torch.tensordot(R[r], B[node], dims=([1], [1]))
            B[node] = torch.tensordot(R[l], B[node], dims=([1], [1]))
            if not dtree.is_root(node):
                # Der Transfertensor der Wurzel muss nicht mehr orthogonalisiert werden
                # Daher wird dieser Abschnitt nur dann durchgefuehrt, falls node ungleich der Wurzel ist
                # Berechne also die QR Zerlegung der Matrizierung des geupdateten Transfertensors
                B[node], R[node] = torch.linalg.qr(self.matricise(B[node], t=(0, 1)), mode="complete")
                # Dematriziere den orthogonalisierten Transfertensor wieder zu 3D
                B[node] = self.dematricise(B[node],
                                                      shape=(R[l].shape[0], R[r].shape[0], B[node].shape[1]), t=(0, 1))
            # Gebe Speicher den Speicher der R-Matrizen der Kinder frei
            del R[l]
            del R[r]
    # Setze die Flag, dass self ein orthogonaler HTucker Tensor ist
    self.is_orthog = True
    return self
