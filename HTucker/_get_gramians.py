import torch


def get_gramians(self):
    """
    Berechnet die reduzierten Gram'schen Matrizen des durch self repraesentierten hierarchischen Tuckertensors.
    Hinweis: Ist self nicht orthogonal, wird es vor der Berechnung der reduzierten Gram'schen Matrizen
    orthogonalisiert.
    :param self: HTucker.HTucker
    :return: dict (tuple: int) -> torch.Tensor
    """
    x = self

    # Orthogonalisiere x, falls notwendig
    if not x.is_orthog:
        x.orthogonalize()
        #warnings.warn("HTucker Tensor {} wurde waehrend der Berechnung der reduzierten Gram'schen Matrizen "
        #              "orthogonalisiert.".format(self))

    # Gramian dict
    G = {x.dtree.get_root(): torch.ones(1, 1)}

    # Traversiere den Dimensionsbaum top down beginnend bei der Wurzel
    # Berechne dabei die jeweiligen reduzierten Gram'schen Matrizen
    for level in range(0, x.dtree.get_depth(), 1):
        for node in x.dtree.get_nodes_of_lvl(level):
            if x.dtree.is_leaf(node):
                # Gram'sche Matrix wurde bereits beim Elternknoten berechnet
                continue
            # Kinder von Node
            l, r = x.dtree.get_children(node)
            # Kontrahiere den Transfertensor von node mit der reduzierten Gramschen'Matrix von node
            BG = torch.tensordot(x.B[node], G[node], dims=([2], [1]))
            # Be
            G[l] = torch.tensordot(x.B[node], BG, dims=([1, 2], [1, 2]))
            G[r] = torch.tensordot(x.B[node], BG, dims=([0, 2], [0, 2]))
    return G
