import torch
from math import sqrt
from copy import deepcopy


def truncate_sum(cls, summands: list, opts: dict):
    """
    Berechnet die Summe der hierarchischen Tuckertensoren aus 'summands'. Die Summe wird dabei en passant entsprechend
    der Constraints in 'opts' gekuerzt.
    Hinweis: Es wird vorausgesetzt, dass die Dimensionsbaeume aller Summanden uebereinstimmen.
    ______________________________________________________________________
    Parameter:
    - summands [HTucker.HTTensor,...]: Die Summanden gegeben als hierarchische Tuckertensoren.
    - opts dict: Enthaelt mindestens eine der folgenden Optionen:
                                    - "max_rank": positiver integer | Legt den maximalen hierarchischen Rang
                                                  fest
                                    - "err_tol_abs": positiver float | Legt die einzuhaltende absolute
                                                     Fehlertoleranz fest
                                    - "err_tol_rel": positiver float | Left die einzuhaltende relative
                                                     Fehlertoleranz fest
    ______________________________________________________________________
    Output:
    (HTucker.HTTensor,): Die Summe gegeben als hierarchischer Tuckertensor.
    ______________________________________________________________________
    Beispiel:
    X, Y, Z = HTTensor.randn((3,4,5,6)), HTTensor.randn((3,4,5,6)), HTTensor.randn((3,4,5,6))
    opts = {"max_rank": 25, "err_tol_abs": 10.0}
    summe = HTTensor.truncate_sum([X,Y,Z], opts)
    """
    # Pruefe die Summanden
    if not isinstance(summands, list):
        raise TypeError("Argument 'summands': type(summands)={} | summands ist keine list.".format(type(summands)))
    if not all(isinstance(item, cls) for item in summands):
        raise TypeError("Argument 'summands': summands enthaelt Elemente, die keine HTucker Tensoren sind.")
    if len(summands) < 2:
        raise ValueError("Argument 'summands': len(summands)={} | summands enthaelt"
                         " weniger als zwei Summanden.".format(len(summands)))
    if not all(summands[0].dtree.is_equal(item.dtree) for item in summands):
        raise ValueError("Argument 'summands': Die Summanden sind nicht kompatibel, da nicht alle Dimensionsbaeume"
                         " uebereinstimmen.")
    # Pruefe das Optionen dicts
    cls._check_opts(opts)

    # Blattmatrixdict, Transfertensordict und Dimtree des resultierenden HTucker Tensors
    U, B, dtree = {}, {}, deepcopy(summands[0].dtree)

    # Anpassen der Fehlertoleranzen in opts
    # Soll global der Fehler err eingehalten werden, muss der Kuerzungsfehler pro Knoten
    # kleiner gleich err / sqrt((Tensorordnung * 2 - 2)) bleiben
    if opts is not None:
        opts = {k: (v / sqrt(summands[0].get_order() * 2 - 2) if k in ["err_tol_abs", "err_tol_rel"]
                    else v) for k, v in opts.items()}

    # Berechne die reduzierten Gram'schen Matrizen der impliziten Summe
    G = cls._get_gramians_sum(summands)

    # Berechne gekuerzte Blattmatrizen der impliziten Summe
    # Update dabei on the fly den Transfertensor des Elternknotens
    B_upd = {node: {} for node in dtree.get_inner_nodes()}
    for leaf in dtree.get_leaves():
        # Konkatenieren der Blattmatrizen
        U_cat = torch.hstack(tuple(item.U[leaf] for item in summands))
        # QR Zerlegung
        Q, R = torch.linalg.qr(U_cat, mode="reduced")
        # Update reduzierte Gram'sche Matrix
        G_upd = R @ G[leaf] @ R.T
        # Berechne davon linke Singulaervektoren
        S, sv = cls.left_svd_gramian(G_upd)
        # Berechne basierend auf opts wie viele Spalten behalten werden
        rank = cls._get_truncation_rank(sv, opts)
        S = S[:, :rank]
        # Berechne schliesslich finale Blattmatrix
        U[leaf] = Q @ S

        # Update Transfertensor des Elternknotens
        # Adaptiere R entsprechend der Kuerzung
        R = S.T @ R
        # Teile dafuer zunaechst R in R=[R1 | R2 | ... | Rn]
        # wobei die Spaltenanzahl von Ri durch den entsprechenden Rang des i-ten Summanden vorgegeben ist
        R = torch.split(R, [item.get_rank()[leaf] for item in summands], dim=1)
        # Multipliziere nun R[i] in den Transfertensor des Elternknotens des i-ten Summanden
        par = dtree.get_parent(leaf)
        for i in range(len(summands)):
            if dtree.is_left(leaf):
                if i in B_upd[par]:
                    B_upd[par][i] = torch.tensordot(R[i], B_upd[par][i], dims=([1], [0]))
                else:
                    B_upd[par][i] = torch.tensordot(R[i], summands[i].B[par], dims=([1], [0]))
            else:
                if i in B_upd[par]:
                    B_upd[par][i] = torch.tensordot(R[i], B_upd[par][i], dims=([1], [1]))
                    B_upd[par][i] = torch.movedim(B_upd[par][i], source=0, destination=1)
                else:
                    B_upd[par][i] = torch.tensordot(R[i], summands[i].B[par], dims=([1], [1]))
                    B_upd[par][i] = torch.movedim(B_upd[par][i], source=0, destination=1)

    # Berechne nun gekuerzte Transfertensoren der impliziten Summe
    # Durschreite den Dimensionsbaum dazu bottom-up
    for level in range(dtree.get_depth()-1, 0, -1):
        for node in dtree.get_nodes_of_lvl(level):
            if dtree.is_leaf(node):
                continue
            # Konkateniere die aktualisierten Transfertensoren der Summanden
            B_cat = torch.cat([B_upd[node][i] for i in range(len(summands))], dim=2)
            # Berechne QR Zerlegung der Matrizierung davon
            Q, R = torch.linalg.qr(cls.matricise(B_cat, t=(0,1)), mode="reduced")
            # Dematriziere Q zu 3D Transfertensor
            Q = cls.dematricise(Q, shape=(B_cat.shape[0], B_cat.shape[1], Q.shape[1]), t=(0,1))

            # Aktualisiere reduzierte Gram'sche Matrix
            G_upd = R @ G[node] @ R.T
            # Berechne davon linke Singulaervektoren
            S, sv = cls.left_svd_gramian(G_upd)
            # Berechne basierend auf opts wie viele Spalten behalten werden
            rank = cls._get_truncation_rank(sv, opts)
            S = S[:, :rank]
            # Berechne schliesslich finalen Transfertensor
            B[node] = torch.tensordot(Q, S, dims=([2], [0]))

            # Update Transfertensor des Elternknotens
            # Adaptiere R entsprechend der Kuerzung
            R = S.T @ R
            # Teile dafuer zunaechst R in R=[R1 | R2 | ... | Rn]
            # wobei die Spaltenanzahl von Ri durch den entsprechenden Rang des i-ten Summanden vorgegeben ist
            R = torch.split(R, [item.get_rank()[node] for item in summands], dim=1)
            # Multipliziere nun R[i] in den Transfertensor des Elternknotens des i-ten Summanden
            par = dtree.get_parent(node)
            for i in range(len(summands)):
                if dtree.is_left(node):
                    if i in B_upd[par]:
                        B_upd[par][i] = torch.tensordot(R[i], B_upd[par][i], dims=([1], [0]))
                    else:
                        B_upd[par][i] = torch.tensordot(R[i], summands[i].B[par], dims=([1], [0]))
                else:
                    if i in B_upd[par]:
                        B_upd[par][i] = torch.tensordot(R[i], B_upd[par][i], dims=([1], [1]))
                        B_upd[par][i] = torch.movedim(B_upd[par][i], source=0, destination=1)
                    else:
                        B_upd[par][i] = torch.tensordot(R[i], summands[i].B[par], dims=([1], [1]))
                        B_upd[par][i] = torch.movedim(B_upd[par][i], source=0, destination=1)
            # Gebe B_upd[node] frei
            del B_upd[node]

    # Wurzelbehandlung..
    B_root = torch.sum(torch.concat([item for item in B_upd[dtree.get_root()].values()], dim=2), dim=2)
    B[dtree.get_root()] = B_root.reshape(B_root.shape[0], B_root.shape[1], 1)

    # Erstelle HTucker Tensor der Summe
    z = cls(U=U, B=B, dtree=dtree, is_orthog=False)

    return z






