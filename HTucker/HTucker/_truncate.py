import torch
from .dimtree import dimtree
from math import sqrt

def truncate(cls, x: torch.Tensor, opts: dict=None):
        """
        Berechnet das hierarchische Tuckerformat des vollen Tensors 'x'. Ist 'opts' None, so wird eine exakte Darstellung
        von 'x' im hierarchischen Tuckerformat berechnet. Ansonsten wird lediglich eine Approximation, die den in
        'opts' hinterlegten Constraints genuegt, berechnet.
        ______________________________________________________________________
        Parameter:
        - x torch.Tensor: Der torch.Tensor dessen hierarchisches Tuckerformat berechnet wird
        - opts dict: Enthaelt mindestens eine der folgenden Optionen:
                                            - "max_rank": positiver integer | Legt den maximalen hierarchischen Rang
                                                          fest
                                            - "err_tol_abs": positiver float | Legt die einzuhaltende absolute
                                                             Fehlertoleranz fest
                                            - "err_tol_rel": positiver float | Left die einzuhaltende relative
                                                             Fehlertoleranz fest
        ______________________________________________________________________
        Output:
        (HTucker.HTTensor,): Das (ranggekuerzte) hierarchische Tuckerformat zu 'x'.
        ______________________________________________________________________
        Beispiel:
        a)
        x = torch.randn(3,4,5,6)
        xh = HTTensor.truncate(x)
        type(xh)    # = HTucker.HTTensor
        b)
        x = torch.randn(10,17,3,4)
        opts = {"err_tol_abs": 10.0}
        xh = HTTensor.truncate(x, opts)
        torch.linalg.norm(x-xh.full())    # < 10.0
        c)
        x = torch.randn(10,17,3,4)
        opts = {"err_tol_rel": 0.01}
        xh = HTTensor.truncate(x, opts)
        torch.linalg.norm(x-xh.full())/torch.linalg.norm(x)    # < 0.01
        d)
        x = torch.randn(10,17,3,4)
        opts = {"max_rank": 15, "err_tol_abs": 10.0}
        xh = HTTensor.truncate(x, opts)
        torch.linalg.norm(x-xh.full())    # ggf. > 10.0, da "max_rank" den maximalen hierarchischen Rang beschraenkt
        """

        # Argumentchecks: x
        if not isinstance(x, torch.Tensor):
            raise TypeError("Argument 'x': type(x)={} | x ist kein torch.Tensor.".format(type(x)))
        if x.dim() < 2:
            raise ValueError("Argument 'x': x.shape={} | x ist kein Tensor von Ordnung 2 oder höher.".format(x.shape))
        # Argumentchecks: opts
        if opts is not None:
            cls._check_opts(opts)

        # Anpassen der Fehlertoleranzen in opts
        # Soll global der Fehler e eingehalten werden, muss der Kuerzungsfehler pro Knoten
        # kleiner gleich e / sqrt((Tensorordnung * 2 - 2)) bleiben
        if opts is not None:
            opts = {k: (v / sqrt(len(x.shape) * 2 - 2) if k in ["err_tol_abs", "err_tol_rel"]
                        else v) for k, v in opts.items()}

        # Initialisierung der Instanzvariablen des zu erzeugenden hierarchischen Tuckertensors
        dtree = dimtree.get_canonic_dimtree(x.dim())
        U = {}
        B = {}

        # Vorbereitung des Kerntensors
        C = None

        # Buchfuehrung ueber die Raenge der Knoten
        rank = {}

        # Berechnung der Blattmatrizen
        # Jedes Blatt t=(n_t,) repraesentiert mit n_t genau eine Dimension
        # Die Blaetter werden absteigender Reihenfolge durchlaufen
        # n_t1 > n_t2 > n_t3 > ... > n_td
        for t in sorted(dtree.get_leaves())[::-1]:
            # Berechnung der Blattmatrix U_t
            x_as_matrix = cls.matricise(x, t)
            U[t], sv = cls.left_svd_qr(x_as_matrix)#,_ = torch.linalg.svd(x_as_matrix, full_matrices=False)# cls.left_svd_qr(...)
            if opts:
                # Rangkuerzung
                U[t] = U[t][:, :cls._get_truncation_rank(sv, opts)]
            # Aktualisierung des rank dicts
            rank[t] = U[t].shape[1]
            # Aktualisierung des Kerntensors C
            if C is None:
                C = torch.tensordot(U[t].T, x, dims=([1], [x.dim() - 1]))
            else:
                C = torch.tensordot(U[t].T, C, dims=([1], [x.dim() - 1]))

        # Berechnung der Transfertensoren
        # Die inneren Knoten werden von unten nach oben durchlaufen
        for level in range(dtree.get_depth()-1, -1, -1):
            # Kopie des aktuellen Kerntensors
            C_new = C.detach()
            # Berechnung des Transfertensors
            for t in dtree.get_nodes_of_lvl(level):
                if dtree.is_leaf(t):
                    continue
                elif dtree.is_root(t):
                    B[t] = cls.matricise(C, t)
                    rank[t] = 1
                else:
                    C_as_matrix = cls.matricise(C, t)
                    B[t], sv = cls.left_svd_qr(C_as_matrix)#, _ = torch.linalg.svd(C_as_matrix, full_matrices=False) # cls.left_svd_qr(...)
                    if opts:
                        # Rangkuerzung
                        B[t] = B[t][:, :cls._get_truncation_rank(sv, opts)]
                    # Aktualisierung des rank dicts
                    rank[t] = B[t].shape[1]
                    # Aktualisierung des Kerntensors
                    new_shape = list(C_new.shape)
                    C_new = cls.matricise(C_new, t)
                    C_new = B[t].T @ C_new
                    left, right = dtree.get_children(t)
                    new_shape[left[-1]] = 1
                    new_shape[right[-1]] = rank[t]
                    C_new = cls.dematricise(C_new, tuple(new_shape), t)
                # Reshape Transfertensor zu 3D
                rank_left_child = rank[dtree.get_left(t)]
                rank_right_child = rank[dtree.get_right(t)]
                B[t] = cls.dematricise(B[t], (rank_left_child, rank_right_child, rank[t]), (0, 1))
            C = C_new
        
        return cls(U=U, B=B, dtree=dtree, is_orthog=True)