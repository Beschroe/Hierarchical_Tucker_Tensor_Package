from copy import deepcopy
import torch


def squeeze(self, singletons_to_remove=None):
    """
    Entfernt die Singleton Dimension/Dimensionen aus singletons_to_remove des hierarchischen Tuckertensors self.
    Hinweis:    - Der Dimensionsbaum ist danach ggf. nicht mehr kanonisch.
                - Ist sdim=None, so werden alle Singleton Dimensionen entfernt
                - Ein HTucker Tensor ist stets von Ordnung >= 2. Entsprechend koennen ggf. nicht alle Singleton
                - Dimensionen entfernt werden
                - Handelt es sich um einen HTucker Tensor mit ausschliesslich Singleton Dimensionen, wird
                  der repraesentierte Skalar zurueckgegeben
    :param self: Htucker.HTucker
    :param singletons_to_remove: list:int oder int
    :return: HTucker.HTucker
    """
    if singletons_to_remove is None:
        singletons_to_remove = [dim for dim in range(len(self.get_shape())) if self.get_shape()[dim] == 1]
        if not singletons_to_remove:
            # In diesem Fall hat self gar keine Singleton Dimensionen
            return self
    if not isinstance(singletons_to_remove, list):
        if isinstance(singletons_to_remove, int):
            singletons_to_remove = [int]
        else:
            raise TypeError("Argument 'singletons_to_remove': type(singletons_to_remove)={} | singletons_to_remove"
                            " ist weder eine Liste mit int Eintraegen"
                            " noch ein einzelner int.".format(type(singletons_to_remove)))
    if len(singletons_to_remove) != len(set(singletons_to_remove)):
        raise ValueError("Argument 'singletons_to_remove': singletons_to_remove enthaelt Duplikate.")
    if any(dim not in range(self.get_order()) for dim in singletons_to_remove):
        raise ValueError("Argument 'singletons_to_remove': singletons_to_remove enthaelt ungueltige Dimensionen.")
    if any(self.get_shape()[dim] != 1 for dim in singletons_to_remove):
        raise ValueError("Argument 'singletons_to_remove': singletons_to_remove enthaelt ungueltige Dimensionen, "
                         "die keine Singleton Dimensionen sind.")

    # z ist ein HTucker Tensor mit ausschliesslich Singleton Dimensionen
    # Entsprechend repraesentiert z einen Skalar. Dieser Skalar wird zurueckgegeben.
    if len(self.get_shape()) - len(singletons_to_remove) == 0:
        return float(self.full())

    # Kontrolliere, dass der resultierende HTucker Tensor mindestens von Ordnung 2 ist
    if len(self.get_shape()) - len(singletons_to_remove) == 1:
        # Die letzte Singleton Dimension wird behalten
        singletons_to_remove = singletons_to_remove[:-1]

    # Bereite den neuen HTucker Tensor vor
    U = deepcopy(self.U)
    B = deepcopy(self.B)
    dtree = deepcopy(self.dtree)

    # z ist ein HTucker Tensor mit ausschliesslich Singleton Dimensionen
    # Entsprechend repraesentiert z einen Skalar. Dieser Skalar wird zurueckgegeben.
    if len(self.get_shape()) - len(singletons_to_remove) == 0:
        return float(self.full())

    # Entferne Singletons
    singleton_node = _get_next_singleton(dtree, singletons_to_remove)
    while singleton_node is not None:

        # Vater- und Geschwisterknoten
        par, sib = dtree.get_parent(singleton_node), dtree.get_sibling(singleton_node)

        # Multipliziere die Blattmatrix von singleton_node in den Transfertensor des Elternknotens
        if dtree.is_left(singleton_node):
            transfer_tensor = torch.tensordot(U[singleton_node], B[par], dims=([1], [0]))
        else:
            transfer_tensor = torch.tensordot(U[singleton_node], B[par], dims=([1], [1]))

        # Entferne die Singleton Dimension aus transfer_tensor
        transfer_tensor = transfer_tensor.reshape(-1, transfer_tensor.shape[2])

        # Verrechne transfer_tensor mit Geschwisterknoten
        if dtree.is_leaf(sib):

            # Gewisterknoten ist selbst ein Blatt
            # transfer_tensor wird mit der Blattmatrix des Geschwisterknotens multipliziert
            mat_times_tt = torch.tensordot(U[sib], transfer_tensor, dims=([1], [0]))

            # Der gemeinsame Elternknoten, der in der naechsten Iteration nun ein Blattknoten sein wird,
            # erhaelt das Berechnete Produkt (mat_times_tt) als Blattmatrix
            U[par] = mat_times_tt

            # Der gemeinsame Elternknoten ist in der naechsten Iteration ein Blattknoten
            dtree.remove_children(par)
            del B[par]

            # Der betrachtete Knoten (singleton_node) und sein Geschwisterknoten werden aus dem
            # Dimensionsbaum entfernt.
            dtree.remove_node(singleton_node)
            dtree.remove_node(sib)
            del U[singleton_node]
            del U[sib]

        else:
            # Geschwisterknoten ist ein innerer Knoten
            # transfer_tensor wird mit dem Transfertensor des Geschwisterknotens kontrahiert
            tt_times_tt = torch.tensordot(B[sib], transfer_tensor, dims=([2], [0]))

            # Der gemeinsame Elterknoten erhaelt das Produkt als Transfertensor
            B[par] = tt_times_tt

            # Der gemeinsame Elternknoten uebernimmt nun die Kinder des Geschwisterknotens
            dtree.set_children(par, dtree.get_children(sib))

            # Der betrachtete Knoten (singleton_node) und sein Geschwisterknoten werden aus dem
            # Dimensionsbaum entfernt.
            dtree.remove_node(singleton_node)
            dtree.remove_node(sib)
            del U[singleton_node]
            del B[sib]
        singleton_node = _get_next_singleton(dtree, singletons_to_remove)

    # Die Strukturen des Dimensionsbaums dtree, des Transfertensordicts B und des Blattmatrixdicts U
    # wurden bereits aktualisiert. Als naechstes muessen die Dimensionen aktualisiert werden.
    # Beispiel: Betrachte folgende Knoten (0,1), (0,) und (1,). Sei (0,) eine Singleton Dimension, die entfernt wurde.
    #           Dann ist aktuell nur noch der gemeinsame Elternknoten (0,1) vorhanden, der aber nur noch eine Dimension
    #           repraesentiert. Im neuen Dimensionsbaum wird dies die Dimension 0 sein. Entsprechend muss (0,1) auf (0,)
    #           abgebildet werden. Dieses Mapping alte Dimension(en) -> neue Dimension(en) wird im Folgenden berechnet
    old2new = {}
    for old_node in dtree.get_nodes():
        new_node = []
        for dim in old_node:
            if dim in singletons_to_remove:
                continue
            cnt_singletons_smaller_dim = len([singleton for singleton in singletons_to_remove if singleton < dim])
            new_node += [dim - cnt_singletons_smaller_dim]
        if new_node:
            old2new[old_node] = tuple(new_node)
        else:
            smallest_dim = min(old_node)
            cnt_singletons_smaller_smallest_dim = len([singleton for singleton in singletons_to_remove if
                                                       singleton < smallest_dim])
            old2new[old_node] = (smallest_dim - cnt_singletons_smaller_smallest_dim,)

    # Anwendung des Knotenmappings auf den Dimensionsbaum
    new_nodes = {old2new[node]: ([] if dtree.is_leaf(node) else
                                 [old2new[dtree.get_left(node)], old2new[dtree.get_right(node)]])
                 for node in dtree.get_nodes()}
    dtree = type(self.dtree)(new_nodes)

    # Anwendung des Knotenmappings auf die Blattmatrixdict
    U = {old2new[k]: v for k, v in U.items()}

    # Anwendung des Knotenmappings auf das Transfertensordict
    B = {old2new[k]: v for k, v in B.items()}

    # Zurueckgeben des neuen HTucker Tensors
    return type(self)(U=U, B=B, dtree=dtree, is_orthog=False)


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
