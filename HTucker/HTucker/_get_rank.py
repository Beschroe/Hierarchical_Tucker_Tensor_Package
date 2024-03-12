
def get_rank(self):
    """
    Gibt den hierarchischen Rang des hierarchischen Tuckertensors 'self' zurueck.
    Der Rang eines Blattes ist die Anzahl die Spaltenanzahl der Blattmatrix
    Der Rang eines inneren Knotens ist die Groesse der dritten Dimension des Transfertensors
    ______________________________________________________________________
    Output:
    (dict,): Das dict enthaelt fuer jeden Knoten des Dimensionsbaums von 'self' einen Eintrag, der dem hierarchischen
             Rang dieses Knotens entspricht.
    ______________________________________________________________________
    Beispiel:
    (Die folgenden Beispiele liefern pro Durchlauf verschiedene Ergebnisse, da HTucker.randn() zufaellige hierarchische
    Raenge generiert.)
    a)
    x = HTTensor.randn((3,4,5,6))
    rank = x.get_rank()    # = {(0, 1, 2, 3): 1, (0,): 2, (1,): 4, (2,): 3, (3,): 3, (0, 1): 7, (2, 3): 3}
    b)
    x = HTTensor.randn((10,12))
    rank = x.get_rank()    # = {(0, 1): 1, (0,): 6, (1,): 4}
    """

    # Die Wurzel hat immer Rang 1
    rank = {self.dtree.get_root() :1}
    for leaf in self.dtree.get_leaves():
        rank[leaf] = self.U[leaf].shape[1]
    for node in self.dtree.get_inner_nodes():
        if self.dtree.is_root(node):
            continue
        rank[node] = self.B[node].shape[2]
    return rank
