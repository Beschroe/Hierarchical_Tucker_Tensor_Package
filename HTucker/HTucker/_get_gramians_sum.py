import torch
from numpy import cumsum

def _get_gramians_sum(cls, summands: list):
    """
    Hinweis: Dies ist eine interne Funktion.
    ______________________________________________________________________
    Berechnet die reduzierten Gram'schen Matrizen der Summe aller in 'summands' enthaltenen hierarchischen
    Tuckertensoren. Die Berechnung findet hierbei statt, ohne die Summe tatsaechlich auszurechnen.

    Sei X ein Tensor und X_t die zugehoerige t-Matrizierung. Ferner sei U_t eine zugehoerige orthogonale Basis
    des Spaltenraums von X_t. Dann erfuellt die reduzierte Gram'sche Matrix G_t folgende Gleichung:
    X_t @ X_t.T = U_t @ G_t @ U_t.T
    ______________________________________________________________________
    Parameter:
    - summands list mit HTucker.HTTensor Eintraegen: Die Summanden von deren Summe die reduzierten Gram'schen Matrizen
                                                    berechnet werden sollen.
    ______________________________________________________________________
    Output:
    (dict,): Das dict enthaelt fuer jeden Knoten des Dimensionsbaums der impliziten Summe die zugehoerige
             reduzierte Gram'sche Matrix.
    ______________________________________________________________________
    Beispiel:
    X, Y, Z = HTTensor.randn((3,4,5,6)), HTucker.randn((3,4,5,6)), HTucker.randn((3,4,5,6))
    G = HTTensor._get_gramians_sum([X,Y,Z])
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

    # Referenzdimensionsbaum
    dtree = summands[0].dtree

    # Berechne M = U.T @ U fuer jeden Knoten
    # Sei n die Anzahl an Summanden, dann ist U.T @ U eine Matrix mit r_1+r_2+...+r_n Zeilen und Spalten
    # Damit entspricht U.T @ U einer Blockmatrix mit n^2 vielen Bloecken. Der Block (i,j) hat dabei die
    # Groesse r_i x r_j
    # Um dieser Situation Rechnung zu tragen, wird M als einfach geschachteltes dict realisiert.
    # Der erste Key gibt an um welchen Knoten es sich handelt, während der zweite (gegeben als Tupel (i,j))
    # bestimmt, welches geordnete Paar betrachtet wird
    M = {}

    # Iteriere ueber alle Blattknoten
    for leaf in dtree.get_leaves():
        U = torch.hstack(tuple(item.U[leaf] for item in summands))
        M_leaf = U.T @ U
        # cum_ranks = [0, r0, r0+r1, r0+r1+r2, ...]
        cum_ranks = cumsum([0] + [item.U[leaf].shape[1] for item in summands])
        # Baue das innere dict M[leaf]
        M[leaf] = {}
        # Iteriere ueber alle geordneten Paare der Summanden
        for i in range(len(summands)):
            for j in range(len(summands)):
                M[leaf][i,j] = M_leaf[cum_ranks[i]:cum_ranks[i + 1], cum_ranks[j]:cum_ranks[j + 1]]

    # Iteriere bottom up ueber alle inneren Knoten
    for level in range(dtree.get_depth()-1, -1, -1):
        for node in dtree.get_nodes_of_lvl(level):
            if dtree.is_leaf(node):
                continue
            # Kinder
            l, r = dtree.get_children(node)
            # Baue das innere dict M[node]
            M[node] = {}
            for i in range(len(summands)):
                for j in range(len(summands)):
                    M_left_times_B = torch.tensordot(M[l][j, i], summands[i].B[node], dims=([1], [0]))
                    M_right_times_B = torch.tensordot(M[r][i, j], summands[j].B[node], dims=([1], [1]))
                    M_right_times_B = torch.movedim(M_right_times_B, source=0, destination=1)
                    M[node][i,j] = torch.tensordot(M_left_times_B, M_right_times_B, dims=([0, 1], [0, 1]))

    # Nachdem nun fuer jeden Knoten M[node] vorhanden ist, koennen die reduzierten Gram'schen Matrizen
    # berechnet werden
    # Das nachstehende dict speichert diese
    # Die reduzierte Gram'sche Matrix der Wurzel ist stets 1
    G = {dtree.get_root(): torch.ones(1,1)}
    # Alle weiteren Eintrage von G sind vorerst dicts, die spaeter zu einer Matrix gemerged werden

    # Iteriere top-down durch den Dimensionsbaum
    for level in range(0, dtree.get_depth(), 1):
        for node in dtree.get_nodes_of_lvl(level):
            if dtree.is_leaf(node):
                continue
            # Kinder
            l, r = dtree.get_children(node)
            # Baue die inneren dicts der Kinder
            G[l], G[r] = {}, {}
            for i in range(len(summands)):
                for j in range(len(summands)):
                    if dtree.is_root(node):
                        # Die reduzierte Gram'sche Matrix der Wurzel ist 1, weswegen die
                        # Transfertensoren B_i und B_j des Paares (i,j) direkt verrechnet werden koennen
                        B_times_G_times_B = torch.tensordot(summands[i].B[node], summands[j].B[node],
                                                            dims=([2], [2]))
                    else:
                        B_times_G = torch.tensordot(summands[i].B[node], G[node][j, i],
                                                    dims=([2], [1]))
                        B_times_G_times_B = torch.tensordot(B_times_G, summands[j].B[node],
                                                            dims=([2], [2]))
                    G[l][i,j] = torch.tensordot(B_times_G_times_B, M[r][i,j],
                                                dims=([1, 3], [0, 1]))
                    G[r][i,j] = torch.tensordot(B_times_G_times_B, M[l][i,j],
                                                dims=([0, 2], [0, 1]))

    # Konkateniere die Bloecke der inneren dicts in G der reduzierten Gram'schen Matrizen zu
    # einer grossen Blockmatrix pro Knoten
    for node in dtree.get_nodes():
        if dtree.is_root(node):
            continue
        G[node] = torch.cat([torch.cat([G[node][i,j]
                                        for j in range(len(summands))], dim=1)
                             for i in range(len(summands))], dim=0)

    # Setze Eintrag der Wurzel
    return G

