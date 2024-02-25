import torch
import numpy as np
from copy import deepcopy
from math import sqrt


def ele_mul(self, y, opts=None):
    """
    Berechnet das elementweise Produkt (Hadamard Produkt) aus self und y.
    Dabei wird en passant eine Rangkuerzung entsprechend opts vorgenommen.
    opts enthaelt folgende Optionen:
                                    - "max_rank": positiver integer | Legt den maximalen hierarchischen Rang
                                                  fest
                                    - "err_tol_abs": positiver float | Legt die einzuhaltende absolute
                                                     Fehlertoleranz fest
                                    - "err_tol_rel": positiver float | Left die einzuhaltende relative
                                                     Fehlertoleranz fest
    :param y: HTucker.HTucker
    :param opts: dict
    """
    if not isinstance(y, type(self)):
        raise TypeError("Argument 'y': type(y)={} | y ist kein HTucker.HTucker Objekt.".format(type(y)))
    if not self.dtree.is_equal(y.dtree):
        raise ValueError("Argument 'y': Der Dimensionsbaum von y ist nicht kompatibel.")
    if self.get_shape() != y.get_shape():
        raise ValueError("Argument 'y': y.shape={} | Die shape von y ist nicht kompatibel zur shape von"
                         "self={}.".format(y.get_shape(), self.get_shape()))
    if opts is not None:
        self.check_opts(opts)

    # Aus Lesbarkeitsgruenden
    x = self
    rx = x.get_rank()
    ry = y.get_rank()

    # Buffer die Transfertensoren aus x und y
    Bx = deepcopy(x.B)
    By = deepcopy(y.B)

    # Blattmatrixdict, Transfertensordict und Dimtree des resultierenden HTucker Tensors
    U, B, dtree = {}, {}, deepcopy(x.dtree)

    # Anpassen der Fehlertoleranzen in opts
    # Soll global der Fehler e eingehalten werden, muss der Kuerzungsfehler pro Knoten
    # kleiner gleich e / sqrt((Tensorordnung * 2 - 2)) bleiben
    if opts is not None:
        opts = {k: (v / sqrt(len(x.get_shape()) * 2 - 2) if k in ["err_tol_abs", "err_tol_rel"]
                    else v) for k, v in opts.items()}

    # Orthogonalisiere x und y
    x.orthogonalize()
    y.orthogonalize()

    # Berechne reduzierten Gram'schen Matrizen
    Gx = x.get_gramians()
    Gy = y.get_gramians()

    # Traversiere der Baum bottom-up
    for level in range(dtree.get_depth(), -1, -1):
        for node in dtree.get_nodes_of_lvl(level):
            if dtree.is_root(node):
                # Knoten ist die Wurzel
                # Fuer die Wurzel, die Rang 1 hat, werden keine Singulaevektoren berechnet
                # Der Transfertensor kann direkt aktualisiert werden
                B[node] = Bx[node] * By[node]

            else:
                # Knoten ist nicht die Wurzel
                # Entsprechend muessen fuer die Rangkuerzung zunaechst die linken
                # Singulaervektoren samt Singulaerwerten bestimmt werden

                # Berechne die linken Singulaerwerte
                Qx, svx = self.left_svd_gramian(Gx[node])
                Qy, svy = self.left_svd_gramian(Gy[node])

                # Konvertiere die Singulaerwerte zu np.ndarray
                svx = np.array(svx)
                svy = np.array(svy)

                # Berechne alle Zweierprodukte aus Singulaerwerten
                sv = svx.reshape((-1, 1)) @ svy.reshape((-1, 1)).T

                # Ordne diese Produkte absteigend
                ind = np.argsort(sv.ravel())[::-1]
                sv_flat_and_ordered = sv.ravel()[ind]

                # Bestimme Kuerzungsrang
                # Hinweis: Der Rang bezieht sich auf alle Zweierprodukte. Der maximale Rang
                #          ist damit gegeben als #Spalten in Qx * #Anzahl Spalten in Qy
                if opts:
                    rank = self.get_truncation_rank(torch.Tensor(sv_flat_and_ordered), opts)
                else:
                    rank = len(sv_flat_and_ordered)

                # Bestimme die Spalten, die mitgenommen werden
                # Hinweis: Da die Indizes aus indx und indy angeben, welche Zweierprodukte mitgenommen werden,
                #          ist es moeglich (sogar wahrscheinlich), dass manche Spalten doppelt auftreten
                indx, indy = np.unravel_index(indices=ind[:rank], shape=sv.shape)
                Qx = Qx[:, indx]
                Qy = Qy[:, indy]


                if dtree.is_leaf(node):
                    # Knoten ist ein Blatt
                    # Update die Blattmatrix
                    U[node] = (x.U[node] @ Qx) * (y.U[node] @ Qy)

                else:
                    # Knoten ist ein inneren Knoten
                    # Update Transfertensor
                    Bx[node] = torch.tensordot(Bx[node], Qx, dims=([2], [0]))
                    By[node] = torch.tensordot(By[node], Qy, dims=([2], [0]))
                    B[node] = Bx[node] * By[node]
                # Aktualisiere den Transfertensor des Elternknotens
                # Hierbei ist es unerheblich, ob node ein Blatt oder
                # innerer Knoten ist
                par = dtree.get_parent(node)
                if dtree.is_left(node):
                    Bx[par] = torch.tensordot(Qx, Bx[par], dims=([0], [0]))
                    By[par] = torch.tensordot(Qy, By[par], dims=([0], [0]))
                else:
                    Bx[par] = torch.tensordot(Qx, Bx[par], dims=([0], [1]))
                    Bx[par] = torch.movedim(Bx[par], source=0, destination=1)
                    By[par] = torch.tensordot(Qy, By[par], dims=([0], [1]))
                    By[par] = torch.movedim(By[par], source=0, destination=1)

            # Entferne die gebufferten Transfertensoreintraege
            if not dtree.is_leaf(node):
                del Bx[node]
                del By[node]

    # Erzeuge resultierenden HTucker Tensor, der das gekuerzte elementweise Produkt
    # darstellt
    z = type(self)(U=U, B=B, dtree=dtree, is_orthog=False)

    return z
