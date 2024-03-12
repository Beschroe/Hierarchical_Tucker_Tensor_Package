import torch
import numpy as np
from copy import deepcopy
from math import sqrt


def ele_mul(self, y, opts: dict=None):
    """
    Berechnet das elementweise Produkt (Hadamard Produkt) der beiden hierarchischen Tuckertensoren 'self' und 'y'.
    Voraussetzung dafuer ist, dass deren Dimensionsbaeume uebereinstimmen. Waehrend der Berechnung wird en passant eine
    Rangkuerzung des Produkts entsprechend der Constraints in 'opts' vorgenommen.
    Hinweis: Da die erhaltene Fehlerschranke eher grosszuegig ausfaellt, kann es sinnvoll sein, im Anschluss
             eine weitere Rangkuerzung vorzunehmen.
    ______________________________________________________________________
    Parameter:
    - y HTucker.HTTensor: Ein hierarchischer Tuckertensor dessen Dimensionsbaum mit dem Dimensionsbaum von 'self'
                        uebereinstimmt.
    - opts dict: Das Optionen-dict kann folgende Constraints enthalten:
                                    - "max_rank": positiver integer | Legt den maximalen hierarchischen Rang
                                                  fest
                                    - "err_tol_abs": positiver float | Legt die einzuhaltende absolute
                                                     Fehlertoleranz fest
                                    - "err_tol_rel": positiver float | Left die einzuhaltende relative
                                                     Fehlertoleranz fest
    ______________________________________________________________________
    Output:
    (HTucker.HTTensor,): Das elementweise Produkt.
    ______________________________________________________________________
    Beispiel:
                  HTucker.HTTensor                   <~~~>          torch.Tensor
    a) Mit en passant Rangkuerzung
       x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
       y = HTTensor.randn((3,4,5,6))             |           y = torch.randn(3,4,5,6)
       opts = {"max_rank": 10,                   |           prod = x * y
               "err_tol_abs": 1e-2}              |
       prod = x.ele_mul(y, opts)                 |

    b) Mit en passant Rangkuerzung
       x = HTTensor.randn((10,8,4))              |           x = torch.randn(10,8,4)
       y = HTTensor.randn((10,8,4))              |           y = torch.randn(10,8,4)
       opts = {"err_tol_abs": 1e-2}              |           prod = x * y
       prod = x.ele_mul(y, opts)                 |

    x) Ohne en passant Rangkuerzung
       x = HTTensor.randn((20,4,3,7))            |           x = torch.randn(20,4,3,7)
       y = HTTensor.randn((20,4,3,7))            |           y = torch.randn(20,4,3,7)
       prod = x.ele_mul(y)                       |           prod = x * y
    """
    if not isinstance(y, type(self)):
        raise TypeError("Argument 'y': type(y)={} | y ist kein HTucker.HTTensor Objekt.".format(type(y)))
    if not self.dtree.is_equal(y.dtree):
        raise ValueError("Argument 'y': Der Dimensionsbaum von y ist nicht kompatibel.")
    if self.get_shape() != y.get_shape():
        raise ValueError("Argument 'y': y.shape={} | Die shape von y ist nicht kompatibel zur shape von"
                         "self={}.".format(y.get_shape(), self.get_shape()))
    if opts is not None:
        self._check_opts(opts)

    # Erzeuge Kopien
    x = deepcopy(self)
    y = deepcopy(y)


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
    Gx = x._get_gramians()
    Gy = y._get_gramians()

    # Traversiere der Baum bottom-up
    for level in range(x.dtree.get_depth(), -1, -1):
        for node in x.dtree.get_nodes_of_lvl(level):
            if x.dtree.is_root(node):
                # Knoten ist die Wurzel
                # Fuer die Wurzel, die Rang 1 hat, werden keine Singulaevektoren berechnet
                # Der Transfertensor kann direkt aktualisiert werden
                x.B[node] = x.B[node] * y.B[node]

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
                    rank = self._get_truncation_rank(torch.Tensor(sv_flat_and_ordered), opts)
                else:
                    rank = len(sv_flat_and_ordered)

                # Bestimme die Spalten, die mitgenommen werden
                # Hinweis: Da die Indizes aus indx und indy angeben, welche Zweierprodukte mitgenommen werden,
                #          ist es moeglich (sogar wahrscheinlich), dass manche Spalten doppelt auftreten
                indx, indy = np.unravel_index(indices=ind[:rank], shape=sv.shape)
                Qx = Qx[:, indx]
                Qy = Qy[:, indy]


                if x.dtree.is_leaf(node):
                    # Knoten ist ein Blatt
                    # Update die Blattmatrix
                    x.U[node] = (x.U[node] @ Qx) * (y.U[node] @ Qy)

                else:
                    # Knoten ist ein inneren Knoten
                    # Update Transfertensor
                    x.B[node] = torch.tensordot(x.B[node], Qx, dims=([2], [0]))
                    y.B[node] = torch.tensordot(y.B[node], Qy, dims=([2], [0]))
                    x.B[node] = x.B[node] * y.B[node]
                # Aktualisiere den Transfertensor des Elternknotens
                # Hierbei ist es unerheblich, ob node ein Blatt oder
                # innerer Knoten ist
                par = x.dtree.get_parent(node)
                if x.dtree.is_left(node):
                    x.B[par] = torch.tensordot(Qx, x.B[par], dims=([0], [0]))
                    y.B[par] = torch.tensordot(Qy, y.B[par], dims=([0], [0]))
                else:
                    x.B[par] = torch.tensordot(Qx, x.B[par], dims=([0], [1]))
                    x.B[par] = torch.movedim(x.B[par], source=0, destination=1)
                    y.B[par] = torch.tensordot(Qy, y.B[par], dims=([0], [1]))
                    y.B[par] = torch.movedim(y.B[par], source=0, destination=1)

    # Setze is_orthog Flag auf false
    x.is_orthog = False
    return x
