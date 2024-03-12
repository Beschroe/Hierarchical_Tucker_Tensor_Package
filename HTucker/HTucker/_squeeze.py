from copy import deepcopy
import torch


def squeeze(self, dims: list | int = None):
    """
    Entfernt Singleton Dimensionen aus dem hierarchischen Tuckertensor 'self'.
    Fall 1) dims == None: Entfernt alle Singleton Dimensionen.
    Fall 2) dims != None: Entfernt alle in dims aufgefuehrten Singleton Dimensionen.
    Hinweis :   - Der Dimensionsbaum ist danach ggf. nicht mehr kanonisch.
                - Ein HTucker Tensor ist stets von Ordnung >= 2. Entsprechend koennen ggf. nicht alle Singleton
                  Dimensionen entfernt werden.
                - Handelt es sich um einen hierarchischen Tensor mit ausschliesslich Singleton Dimensionen, wird der
                  hierarchische Tuckertensor entfaltet und der erhaltene Skalar zurueckgegeben.
    ______________________________________________________________________
    Parameter:
    - dims [int,...] | int: Die zu entfernenden Singleton Dimensionen.
    ______________________________________________________________________
    Output:
    (HTucker.HTTensor,) Der gesqueezte hierarchische Tuckertensor.
    ______________________________________________________________________
    Beispiel:
                      HTucker.HTTensor         <~~~>          torch.Tensor
    a)
       x = HTTensor.randn((3,4,1,6))             |           x = torch.randn(3,4,1,6)
       x.squeeze()                               |           x.squeeze()
       x.get_shape    # = (3,4,6)                |           x.shape    # = torch.size([3,4,6])

    b)
       x = HTTensor.randn((1,4,1,6))             |           x = torch.randn(1,4,1,6)
       x.squeeze()                               |           x.squeeze()
       x.get_shape    # = (4,6)                  |           x.shape    # = torch.size([4,6])

    c)
       x = HTTensor.randn((1,4,1,6))             |           x = torch.randn(1,4,1,6)
       x.squeeze(dim=2)                          |           x.squeeze(dim=2)
       x.get_shape    # = (1,4,6)                |           x.shape    # = torch.size([1,4,6])

    d)
       x = HTTensor.randn((1,4,1,6,5,1))          |           x = torch.randn(1,4,1,6,5,1)
       x.squeeze(dim=[0,5])                       |           x.squeeze(dim=[0,5])
       x.get_shape    # = (4,1,6,5)               |           x.shape    # = torch.size([4,1,6,5])
    """

    if dims is None:
        dims = [dim for dim in range(len(self.get_shape())) if self.get_shape()[dim] == 1]
        if not dims:
            # In diesem Fall hat self gar keine Singleton Dimensionen
            return self
    if not isinstance(dims, list):
        if isinstance(dims, int):
            dims = [dims]
        else:
            raise TypeError("Argument 'dims': type(dims)={} | dims"
                            " ist weder eine Liste mit int Eintraegen"
                            " noch ein einzelner int.".format(type(dims)))
    if len(dims) != len(set(dims)):
        raise ValueError("Argument 'dims': dims enthaelt Duplikate.")
    if any(dim not in range(self.get_order()) for dim in dims):
        raise ValueError("Argument 'dims': dims enthaelt ungueltige Dimensionen.")
    if any(self.get_shape()[dim] != 1 for dim in dims):
        raise ValueError("Argument 'dims': dims enthaelt ungueltige Dimensionen, "
                         "die keine Singleton Dimensionen sind.")

    # z ist ein HTucker Tensor mit ausschliesslich Singleton Dimensionen
    # Entsprechend repraesentiert z einen Skalar. Dieser Skalar wird zurueckgegeben.
    if len(self.get_shape()) - len(dims) == 0:
        return float(self.full())

    # Kontrolliere, dass der resultierende HTucker Tensor mindestens von Ordnung 2 ist
    if len(self.get_shape()) - len(dims) == 1:
        # Die letzte Singleton Dimension wird behalten
        dims = dims[:-1]

    # Fuer bessere Lesbarkeit
    x = self

    # z ist ein HTucker Tensor mit ausschliesslich Singleton Dimensionen
    # Entsprechend repraesentiert z einen Skalar. Dieser Skalar wird zurueckgegeben.
    if len(self.get_shape()) - len(dims) == 0:
        return float(self.full())

    # Entferne Singletons
    singleton_node = _get_next_singleton(x.dtree, dims)
    while singleton_node is not None:

        # Vater- und Geschwisterknoten
        par, sib = x.dtree.get_parent(singleton_node), x.dtree.get_sibling(singleton_node)

        # Multipliziere die Blattmatrix von singleton_node in den Transfertensor des Elternknotens
        if x.dtree.is_left(singleton_node):
            transfer_tensor = torch.tensordot(x.U[singleton_node], x.B[par], dims=([1], [0]))
        else:
            transfer_tensor = torch.tensordot(x.U[singleton_node], x.B[par], dims=([1], [1]))

        # Entferne die Singleton Dimension aus transfer_tensor
        transfer_tensor = transfer_tensor.reshape(-1, transfer_tensor.shape[2])

        # Verrechne transfer_tensor mit Geschwisterknoten
        if x.dtree.is_leaf(sib):

            # Gewisterknoten ist selbst ein Blatt
            # transfer_tensor wird mit der Blattmatrix des Geschwisterknotens multipliziert
            mat_times_tt = torch.tensordot(x.U[sib], transfer_tensor, dims=([1], [0]))

            # Der gemeinsame Elternknoten, der in der naechsten Iteration nun ein Blattknoten sein wird,
            # erhaelt das Berechnete Produkt (mat_times_tt) als Blattmatrix
            x.U[par] = mat_times_tt

            # Der gemeinsame Elternknoten ist in der naechsten Iteration ein Blattknoten
            x.dtree.remove_children(par)
            del x.B[par]

            # Der betrachtete Knoten (singleton_node) und sein Geschwisterknoten werden aus dem
            # Dimensionsbaum entfernt.
            x.dtree.remove_node(singleton_node)
            x.dtree.remove_node(sib)
            del x.U[singleton_node]
            del x.U[sib]

        else:
            # Geschwisterknoten ist ein innerer Knoten
            # transfer_tensor wird mit dem Transfertensor des Geschwisterknotens kontrahiert
            tt_times_tt = torch.tensordot(x.B[sib], transfer_tensor, dims=([2], [0]))

            # Der gemeinsame Elterknoten erhaelt das Produkt als Transfertensor
            x.B[par] = tt_times_tt

            # Der gemeinsame Elternknoten uebernimmt nun die Kinder des Geschwisterknotens
            x.dtree.set_children(par, x.dtree.get_children(sib))

            # Der betrachtete Knoten (singleton_node) und sein Geschwisterknoten werden aus dem
            # Dimensionsbaum entfernt.
            x.dtree.remove_node(singleton_node)
            x.dtree.remove_node(sib)
            del x.U[singleton_node]
            del x.B[sib]
        singleton_node = _get_next_singleton(x.dtree, dims)

    # Die Strukturen des Dimensionsbaums dtree, des Transfertensordicts B und des Blattmatrixdicts U
    # wurden bereits aktualisiert. Als naechstes muessen die Dimensionen aktualisiert werden.
    # Beispiel: Betrachte folgende Knoten (0,1), (0,) und (1,). Sei (0,) eine Singleton Dimension, die entfernt wurde.
    #           Dann ist aktuell nur noch der gemeinsame Elternknoten (0,1) vorhanden, der aber nur noch eine Dimension
    #           repraesentiert. Im neuen Dimensionsbaum wird dies die Dimension 0 sein. Entsprechend muss (0,1) auf (0,)
    #           abgebildet werden. Dieses Mapping alte Dimension(en) -> neue Dimension(en) wird im Folgenden berechnet
    old2new = {}
    for old_node in x.dtree.get_nodes():
        new_node = []
        for d in old_node:
            if d in dims:
                continue
            cnt_singletons_smaller_dim = len([singleton for singleton in dims if singleton < d])
            new_node += [d - cnt_singletons_smaller_dim]
        if new_node:
            old2new[old_node] = tuple(new_node)
        else:
            smallest_dim = min(old_node)
            cnt_singletons_smaller_smallest_dim = len([singleton for singleton in dims if
                                                       singleton < smallest_dim])
            old2new[old_node] = (smallest_dim - cnt_singletons_smaller_smallest_dim,)

    # Anwendung des Knotenmappings auf den Dimensionsbaum
    new_nodes = {old2new[node]: ([] if x.dtree.is_leaf(node) else
                                 [old2new[x.dtree.get_left(node)], old2new[x.dtree.get_right(node)]])
                 for node in x.dtree.get_nodes()}
    x.dtree.nodes = new_nodes

    # Anwendung des Knotenmappings auf die Blattmatrixdict
    x.U = {old2new[k]: v for k, v in x.U.items()}

    # Anwendung des Knotenmappings auf das Transfertensordict
    x.B = {old2new[k]: v for k, v in x.B.items()}

    return x


def _get_next_singleton(dtree, sdim):
    """
    Gibt den Knoten einer Singleton Dimension zurueck, wobei Knoten, deren Geschwister ebenfalls Singletons sind,
    bevorzugt werden
    :param dtree: dimtree.dimtree
    :return: tuple:int
    """
    singletons = []
    for leaf in sorted(dtree.get_leaves()):
        if all(item in sdim for item in leaf):
            # leaf ist Singleton
            singletons += [leaf]
    for single in singletons:
        sibling = dtree.get_sibling(single)
        if sibling in singletons:
            return single
    if singletons:
        return singletons[0]
