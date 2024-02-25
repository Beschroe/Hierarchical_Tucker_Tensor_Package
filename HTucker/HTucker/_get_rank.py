

def get_rank(self):
    """
    Gibt den hierarchischen Rang als dict zurueck: self.get_rank()[node] = Rang des Knoten node
    Der Rang eines Blattes entspricht der Spaltenanzahl der zugehoerigen Blattmatrix
    Der Rang eines inneren Knotens ist die Groesse der dritten Dimension des Transfertensors
    """
    # Die Wurzel hat immer Rang 1
    rank = {self.dtree.get_root():1}
    for leaf in self.dtree.get_leaves():
        rank[leaf] = self.U[leaf].shape[1]
    for node in self.dtree.get_inner_nodes():
        if self.dtree.is_root(node):
            continue
        rank[node] = self.B[node].shape[2]
    return rank