from copy import deepcopy
import numpy as np


class dimtree:
    """
    Implementiert einen Dimensionsbaum zur Verwaltung der Dimensionshierarchie eines hierarchischen Tuckertensors.
    """

    def __init__(self, nodes: dict):
        """
        Erzeugt einen Dimensionsbaum auf Grundlage der Knotenhierarchie in nodes.
        :param nodes: dict mit Eintraegen der Form (0,1): [(0,), (1,)]
        """
        dimtree._check_nodes(nodes)
        self.nodes = nodes

    @staticmethod
    def _check_nodes(nodes):
        """
        Prueft, ob nodes einen gueltigen Dimensionsbaum repraesentiert.
        Hinweis: Die Ueberpruefungen sind unvollstaendig.
        :param nodes: dict mit Eintraegen der Form (0,1): [(0,), (1,)]
        :return:
        """
        # Typechecks
        if not isinstance(nodes, dict):
            raise TypeError("Argument 'nodes' = {}: type('nodes')={} ist kein dict.".format(nodes, type(nodes)))
        # Pruefung aller keys und values
        for k, v in nodes.items():
            # Pruefung des keys (Elternknoten)
            if not isinstance(k, tuple):
                raise TypeError("Argument 'nodes' = {}: Der Key {} ist kein tuple: {}.".format(nodes, k, type(k)))
            if not all(np.issubdtype(type(item), np.integer) for item in k):
                raise TypeError("Argument 'nodes' = {}: Der Key {} ist ein tuple mit nicht-integer"
                                " Elementen: {}.".format(nodes, k, tuple(type(item) for item in k)))
            if not all(item >= 0 for item in k):
                raise ValueError("Argument 'nodes' = {}: Der Key {} ist ein tuple mit Elementen < 0.".format(nodes, k))
            # Pruefung des values (Kinderknoten)
            if not isinstance(v, list):
                raise TypeError("Argument 'nodes' = {}: Der Value {} ist keine list: {}".format(nodes, v, type(v)))
            if v:
                # v ist nicht die leere Liste
                # Der durch den Key repraesentierte Knoten hat also Kinder
                if not len(v) == 2:
                    raise ValueError("Argument 'nodes' = {}: Der Value {} ist eine Liste der Laenge {}."
                                     " Gueltig ist eine Laenge von 0 oder 2.".format(nodes, v, len(v)))
                if not all(isinstance(item, tuple) for item in v):
                    raise TypeError("Argument 'nodes' = {}: Der Value {} ist eine Liste mit nicht-tuple"
                                    " Elementen: {}.".format(nodes, v, [type(item) for item in v]))
                if not all(np.issubdtype(type(item), np.integer) for subtuple in v for item in subtuple):
                    raise TypeError("Argument 'nodes' = {}: Der Value {} enthaelt tuple mit nicht-integer"
                                    " Elementen: {}".format(nodes, v,
                                                            [tuple(type(item) for item in subtuple) for subtuple in v]))
                if any(item < 0 for subtuple in v for item in subtuple):
                    raise ValueError(
                        "Argument 'nodes' = {}: Der Value {} enthaelt tuple mit Elementen < 0.".format(nodes, v))
                if any(len(set(subtuple)) != len(subtuple) for subtuple in v):
                    raise ValueError(
                        "Argument 'nodes' = {}: Der Value {} enthaelt tuple mit Duplikaten.".format(nodes, v))
                if len(set(v[0]).intersection(v[1])) > 0:
                    raise ValueError(
                        "Argument 'nodes' = {}: Der Value {} enthaelt tuple mit geteilten Dimensionen.".format(nodes,
                                                                                                               v))
                if k != v[0] + v[1]:
                    raise ValueError("Argument 'nodes' = {} : Der Key {} ist ungleich der Kontaktenation"
                                     " der tuple des Values {}.".format(nodes, k, v))
            else:
                # v ist die leere Liste
                # Der durch den Key repraesentierte Knoten ist also ein Blattknoten
                if len(k) != 1:
                    raise ValueError(
                        "Argument 'nodes' = {}: Der Key {} ist ein tuple mit nicht genau einem Element, obwohl"
                        " der Key ein Blattknoten ist.".format(nodes, k))

    @staticmethod
    def get_canonic_dimtree(nr_dims):
        """
        Erzeugt den kanonischen Dimensionsbaum fuer die Anzahl 'nr_dims' uebergebener Dimensionen.
        :param nr_dims: positiver integer
        :return: dimtree
        """
        if not np.issubdtype(type(nr_dims), np.integer):
            raise TypeError("Argument 'nr_dims' = {}: {} ist kein integer.".format(nr_dims, type(nr_dims)))
        if nr_dims <= 0:
            raise ValueError("Argument 'nr_dims' = {}: nr_dims ist kein positiver integer.".format(nr_dims))
        nodes = {}
        dims = [tuple(range(nr_dims))]
        while dims:
            dim = dims.pop(0)
            if len(dim) > 1:
                # Innerer Knoten
                r = int(np.ceil(len(dim) / 2))
                left, right = dim[:r], dim[r:]
                nodes[dim] = [left, right]
                dims += [left, right]
            else:
                # Blattknoten
                nodes[dim] = []

        # Konstruktoraufruf
        return dimtree(nodes=nodes)

    def get_nr_nodes(self):
        """
        Gibt die Anzahl an Knoten zurueck.
        :return: integer
        """
        return len(self.get_nodes())

    def get_nr_dims(self):
        """
        Gibt die Anzahl an Dimensionen zurueck.
        :return: integer
        """
        if len(self.get_root()) != len(self.get_leaves()):
            raise RuntimeError("Defekter Dimensionsbaum: Die Anzahl an Blattknoten"
                               " ist ungleich der Anzahl an Dimensionen der Wurzel:"
                               " {} != {}".format(len(self.get_root()), len(self.get_leaves())))
        return len(self.get_root())

    def get_depth(self):
        """
        Gibt die Tiefe des Dimensionsbaums zurueck.
        :return: integer
        """
        depth = 0
        for node in self.get_nodes():
            node_lvl = self.get_lvl(node)
            if node_lvl > depth:
                depth = node_lvl
        return depth

    def is_leaf(self, node):
        """
        Gibt True zurueck, falls node ein Blattknoten ist. Ansonsten wird False zurueckgegeben.
        :param node: Ein tuple mit integer Eintraegen
        :return: bool
        """
        # Typecheck
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuecheck
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if self.nodes[node]:
            return False
        else:
            return True

    def get_children(self, node):
        """
        Gibt die Kinder des Knotens node zurueck. Ist node ein Blattknoten, so wird die leere Liste zurueckgegeben.
        :param node: Ein tuple mit integer Eintraegen
        :return: list
        """
        # Typecheck
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuecheck
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        return self.nodes[node]

    def get_root(self):
        """
        Gibt den Wurzelknoten zurueck.
        :return: Ein tuple mit integer Eintraegen
        """
        root = [node for node in self.get_nodes() if self.is_root(node)]
        if len(root) > 1:
            raise RuntimeError("Defekter Dimensionsbaum: Mehrere Wurzeln sind vorhanden: {}".format(root))
        elif len(root) == 0:
            raise RuntimeError("Defekter Dimensionsbaum: Der Dimensionsbaum besitzt keine Wurzel.")
        return root[0]

    def get_lvl(self, node):
        """
        Gibt das Level des Knotens node zurueck.
        :param node: Ein tuple mit integer Eintraegen
        :return: integer
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))

        lvl = 0
        node = self.get_parent(node)
        while node is not None:
            lvl += 1
            node = self.get_parent(node)
        return lvl

    def get_parent(self, node):
        """
        Gibt den Elternknoten des Knotens node zurueck.
        :param node: Ein tuple mit integer Eintraegen
        :return: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))

        parent = [k for k, v in self.nodes.items() if node in v]
        if len(parent) > 1:
            raise RuntimeError("Dimensionsbaum defekt: Der Knoten {} hat mehrere Eltern: {}.".format(node, parent))
        elif len(parent) == 1:
            return parent[0]
        else:
            # Im Fall der Wurzel
            return None

    def print(self):
        """
        Printet den Dimensionsbaum.
        :return: None
        """
        # Berechne Breite der Ausgabe: Anzahl an Zeichen pro Zeile
        # Dazu wird angenommen, dass es sich um einen vollstaendigen Binaerbaum handelt
        longest_leaf = max([len(str(leaf)) for leaf in self.get_leaves()])
        width = (2 ** self.get_depth()) * (longest_leaf + 1)

        def center(token, nr_of_characters):
            # Platziert ein Token in einem String der Laenge nr_of_characters
            assert len(token) <= nr_of_characters
            nr_characters_left = int(np.ceil((nr_of_characters - len(token)) / 2))
            nr_characters_right = int(np.floor((nr_of_characters - len(token)) / 2))
            return " " * nr_characters_left + token + " " * nr_characters_right

        def get_column_index_mapping(node_, ind=0):
            # Berechnet ein Mapping, das zu jedem Knoten einen entsprechenden Spaltenindex bereithaelt.
            if self.is_root(node_):
                if self.is_leaf(node_):
                    # Hat der Baum nur einen Knoten ist die Wurzel gleichzeitg Blatt
                    return {node_: 0}
                cl, cr = self.get_children(node_)
                return {node_: 0} | get_column_index_mapping(cl, 0) | get_column_index_mapping(cr, 0)
            elif self.is_left(node_):
                # Linkes Kind
                if self.is_leaf(node_):
                    return {node_: ind * 2}
                else:
                    cl, cr = self.get_children(node_)
                    return {node_: ind * 2} | get_column_index_mapping(cl, ind * 2) \
                        | get_column_index_mapping(cr, ind * 2)
            else:
                # Rechtes Kind
                if self.is_leaf(node_):
                    return {node_: ind * 2 + 1}
                else:
                    cl, cr = self.get_children(node_)
                    return {node_: ind * 2 + 1} | get_column_index_mapping(cl, ind * 2 + 1) \
                        | get_column_index_mapping(cr, ind * 2 + 1)

        # Erzeuge Ausgabe Matrix
        # Die Zeilen der Matrix entsprechen den Ausgabezeilen
        # Die Spalten einer Zeile entsprechen den jeweiligen Knoten
        # Initialisiert wird die Matrix mit Whitespace Strings, deren Laenge vom jeweiligen Level abhaengt
        mat = [[]] * (self.get_depth() + 1)
        for lvl in range(self.get_depth() + 1):
            row = [" " * (width // (2 ** lvl))] * (2 ** lvl)
            mat[lvl] = row
        # Befuelle Ausgabe Matrix
        mapping = get_column_index_mapping(self.get_root())
        for node, idx in mapping.items():
            lvl = self.get_lvl(node)
            mat[lvl][idx] = center(str(node), len(mat[lvl][idx]))
        for row in mat:
            print("".join(row))

    def get_nodes_dfs(self, node=None, order="nlr"):
        """
        Gibt die Knoten des Subtrees beginnend bei node in depth-first-search Reihenfolge zurueck. Der Parameter order
        gibt dabei an, in welcher Reihenfolge die Kinder eines Knotens und der Knoten selbst behandelt werden.
        :node: Ein tuple mit integer Eintraegen
        :order: Eines der folgenden str Objekte ["nlr", "nrl", "lnr", "lrn", "rnl", "rln"]
        :return: list
        """
        if node is None:
            node = self.get_root()
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        if not all(np.issubdtype(type(item), np.integer) for item in node):
            raise TypeError("Argument 'node' = {}: node enthaelt nicht-integer Eintraege.".format(node))
        if not isinstance(order, str):
            raise TypeError("Argument 'order' = {}: {} ist kein string.".format(order, type(order)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: node ist kein Knoten des Dimensionsbaums.".format(node))
        if order not in ["nlr", "nrl", "lnr", "lrn", "rnl", "rln"]:
            raise ValueError("Argument 'order' = {}: order entspricht keiner der folgenden "
                             "Moeglichkeiten {}.".format(order, ["nlr", "nrl", "lnr", "lrn", "rnl", "rln"]))

        if self.is_leaf(node):
            # Fall: node ist Blattknoten
            return [node]
        else:
            # Fall: node ist innerer Knoten
            cl, cr = self.get_children(node)
            if order == "nlr":
                return [node] + self.get_nodes_dfs(cl, order) + self.get_nodes_dfs(cr, order)
            elif order == "nrl":
                return [node] + self.get_nodes_dfs(cr, order) + self.get_nodes_dfs(cl, order)
            elif order == "lnr":
                return self.get_nodes_dfs(cl, order) + [node] + self.get_nodes_dfs(cr, order)
            elif order == "lrn":
                return self.get_nodes_dfs(cl, order) + self.get_nodes_dfs(cr, order) + [node]
            elif order == "rnl":
                return self.get_nodes_dfs(cr, order) + [node] + self.get_nodes_dfs(cl, order)
            else:
                # order == "rln"
                return self.get_nodes_dfs(cr, order) + self.get_nodes_dfs(cl, order) + [node]

    def get_nodes(self):
        """
        Gibt alle Knoten des Dimensionsbaums zurueck.
        :return: Eine list mit tuple Elementen. Die tuple Elemente haben integer Eintraege.
        """
        return list(self.nodes.keys())

    def get_nodes_bfs(self):
        """
        Gibt alle Knoten des Dimensionsbaums in bfs Reihenfolge zurueck.
        :return: Eine list mit tuple Elementen. Jedes tuple Element hat integer Eintraege
        """
        nodes_to_visit = [self.get_root()]
        nodes_visited = []
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            nodes_visited += [node]
            if self.is_inner(node):
                nodes_to_visit += self.get_children(node)
        return nodes_visited

    def get_nodes_of_lvl(self, lvl):
        """
        Gibt eine Liste aller Knoten auf Level lvl zurueck
        :param lvl: integer
        :return: Eine list mit tuple Elementen. Jedes tuple Element hat integer Eintraege
        """
        # Typecheck
        if not np.issubdtype(type(lvl), np.integer):
            raise TypeError("Argument 'lvl' = {}: {} ist kein integer.".format(lvl, type(lvl)))
        # Valuecheck
        if lvl not in range(self.get_depth() + 1):
            raise ValueError("Argument 'lvl' = {}: Es existiert kein entsprechendes Level.".format(lvl))
        return [node for node in self.get_nodes() if self.get_lvl(node) == lvl]

    def get_leaves(self):
        """
        Gibt eine Liste mit allen Blattknoten zurueck.
        :return: Eine list mit tuple Elementen. Jedes tuple Element hat integer Eintraege
        """
        return [node for node in self.get_nodes() if self.is_leaf(node)]

    def get_inner_nodes(self):
        """
        Gibt die inneren Knoten des Dimensionsbaums zurueck.
        :return: Eine list mit tuple Elementen. Jedes tuple Element hat integer Eintraege
        """
        return [node for node in self.get_nodes() if self.is_inner(node)]

    def is_root(self, node):
        """
        True, falls node der Wurzelknoten ist. Ansonsten False.
        :return: bool
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        parent = self.get_parent(node)
        if parent is None:
            # Knoten hat keinen Elternknoten und muss daher die Wurzel sein
            return True
        else:
            return False

    def is_inner(self, node):
        """
        True, falls node ein innerer Knoten ist. Ansonsten False.
        :param node: Ein tuple mit integer Eintraegen.
        :return: bool
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if self.get_children(node):
            return True
        else:
            return False

    def is_left(self, node):
        """
        True, falls node ein linkes Kind ist. Ansonsten False.
        :param node: Ein tuple mit integer Eintraegen
        :return:
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if self.is_root(node):
            raise ValueError("Argument 'node' = {}: Es handelt sich um den Wurzelknoten,"
                             " der weder linkes noch rechtes Kind ist.".format(node))
        cl, cr = self.get_children(self.get_parent(node))
        if node == cl:
            return True
        elif node == cr:
            return False
        else:
            raise RuntimeError("Defekter Dimensionsdaum: Der Knoten {} kann weder als"
                               " linkes noch als rechtes Kind identifiziert werden.".format(node))

    def is_right(self, node):
        """
        True, falls node ein rechtes Kind ist. Ansonsten False.
        :param node: Ein tuple mit integer Eintraegen
        :return: bool
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if self.is_root(node):
            raise ValueError("Argument 'node' = {}: Es handelt sich um den Wurzelknoten,"
                             " der weder linkes noch rechtes Kind ist.".format(node))
        cl, cr = self.get_children(self.get_parent(node))
        if node == cr:
            return True
        elif node == cl:
            return False
        else:
            raise RuntimeError("Defekter Dimensionsdaum: Der Knoten {} kann weder als"
                               " linkes noch als rechtes Kind identifiziert werden.".format(node))

    def get_right(self, node):
        """
        Gibt das rechte Kind zurueck, falls es sich um einen inneren Knoten handelt. Ansonsten None.
        :param node: Ein tuple mit integer Eintraegen
        :return: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if self.is_inner(node):
            ch = self.get_children(node)
            if len(ch) != 2:
                raise RuntimeError("Defekter Dimensionsbaum: Der Knoten {} hat"
                                   " ungleich 2 Kinder. Kinder: {}.".format(node, ch))
            else:
                return ch[1]
        else:
            return None

    def get_left(self, node):
        """
        Gibt das linke Kind zurueck, falls es sich um einen inneren Knoten handelt. Ansonsten None.
        :param node: Ein tuple mit integer Eintraegen
        :return: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if self.is_inner(node):
            ch = self.get_children(node)
            if len(ch) != 2:
                raise RuntimeError("Defekter Dimensionsbaum: Der Knoten {} hat"
                                   " ungleich 2 Kinder. Kinder: {}.".format(node, ch))
            else:
                return ch[0]
        else:
            return None

    def get_sibling(self, node):
        """
        Gibt den Geschwisterknoten des Knotens 'node' zurueck.
        :param node: Ein tuple mit integer Eintraegen
        :return: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if self.is_root(node):
            raise ValueError(
                "Argument 'node' = {}: Es handelt sich um den Wurzelknoten - dieser hat keine Geschwister.".format(
                    node))
        parent = self.get_parent(node)
        l, r = self.get_children(parent)
        if self.is_left(node):
            return r
        else:
            return l

    def remove_children(self, node):
        """
        Entfernt die Kinder des Knotens 'node', sofern dieser kein Blattknoten ist.
        :param node: Ein tuple mit integer Eintraegen
        :return:
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        self.nodes[node] = []
        return

    def remove_node(self, node):
        """
        Entfernt den Knoten node aus dem Dimensionsbaum, sofern vorhanden.
        :param node: Ein tuple mit integer Eintraegen
        :return:
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Adaptiere Knotenhierarchie
        new_nodes = {}
        for k, v in self.nodes.items():
            if k != node:
                new_nodes[k] = [val for val in v if val != node]
        self.nodes = new_nodes
        return

    def set_children(self, node, children):
        """
        Der Knoten node erhaelt die beiden in children enthaltenen Knoten als Kinder.
        :param node: Ein tuple mit integer Eintraegen
        :param children: Eine list mit tuple Elementen. Jedes dieser tuple hat integer Eintraege
        :return:
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        if not isinstance(children, list):
            raise TypeError("Argument 'children': type(children)={} | children ist keine list.".format(type(children)))
        # Valuechecks
        if any(child not in self.get_nodes() for child in children):
            raise ValueError("Argument 'children': children enthaelt ungueltige Knoten.")
        if len(children) != 2:
            raise ValueError("Argument 'children': len(children)={} | children enthaelt"
                             " ungleich 2 Elemente.".format(len(children)))
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))

        self.nodes[node] = children
        return

    def is_equal(self, dimtree_two):
        """
        Prueft, ob die beiden Dimensionsbaeume self und dimtree_two uebereinstimmen
        :param dimtree_two: Ein dimtree Objekt.
        :return: bool
        """
        if not isinstance(dimtree_two, dimtree):
            raise TypeError("Argument 'dimtree_two': type(dimtree_two) |"
                            " dimtree_two ist kein Dimensionsbaum".format(type(dimtree_two)))
        return self.nodes == dimtree_two.nodes

    def get_minimal_subtree_covering_dims(self, dims: list):
        """
        Gibt die minimale Anzahl an Knoten zurueck, die alle in dims enthaltenen Dimensionen repraesentieren.
        :param dims: Eine list mit integer Eintraegen
        """
        if not isinstance(dims, list):
            raise TypeError("Argument 'dims': type(dims)={} | dims ist keine list.".format(type(dims)))
        if len(dims) == 0:
            return []
        if not set(dims) <= set(self.get_root()):
            raise ValueError("Argument 'dims': Die in in dims enthaltenen Dimensionen sind (teilweise) nicht durch"
                             "den Dimensionsbaum repraesentiert.")
        return self._get_minimal_subtree_covering_dims_rec(self.get_root(), dims)

    def _get_minimal_subtree_covering_dims_rec(self, node: tuple, dims: list):
        """
        Rekursive Helferfunktion fuer get_minimal_nodes_covering_dims.
        """
        if self.is_leaf(node):
            return [node]
        else:
            l, r = self.get_children(node)
            if set(dims) <= set(l):
                # linker Subtree enthaelt alle Dimensionen
                return self._get_minimal_subtree_covering_dims_rec(l, dims)
            elif set(dims) <= set(r):
                # rechter Subtree enthaelt alle Dimensionen
                return self._get_minimal_subtree_covering_dims_rec(r, dims)
            elif set(l).intersection(set(dims)) and set(r).intersection(set(dims)):
                # rechter und linker Subtree enthalten jeweils einen Teil der Dimensionen
                return self._get_minimal_subtree_covering_dims_rec(l, dims) + [node] + \
                    self._get_minimal_subtree_covering_dims_rec(r, dims)
            elif set(l).intersection(set(dims)) and not set(r).intersection(set(dims)):
                # Nur linker Subtree enthaelt Teil der Dimensionen
                return self._get_minimal_subtree_covering_dims_rec(l, dims) + [node]
            elif not set(l).intersection(set(dims)) and set(r).intersection(set(dims)):
                # Nur rechter Subtree enthaelt Teil der Dimensionen
                return [node] + self._get_minimal_subtree_covering_dims_rec(r, dims)
            else:
                # Weder linker noch rechter Subtree enthalten mindestens einen Teil der Dimensionen
                # Dieser Fall tritt bei korrekter Funktionsweise nicht ein
                raise RuntimeError("Die Kinder des Knotens {} gegeben als {} und {} enthalten"
                                   "keine der Dimensionen {}. Dieser Fall darf bei korrekter Funktionsweise"
                                   "nicht auftreten.".format(node, l, r, dims))

    def get_subtree(self, node: tuple):
        """
        Gibt den subtree beginnend bei node zurueck.
        :param node: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))

        nodes = self.get_nodes_dfs(node)
        children = {}
        for nd in nodes:
            children[nd] = self.get_children(nd)

        return type(self)(children)

    def remove_subtree(self, node: tuple):
        """
        Entfernt den subtree, dessen Wurzel dem Knoten node entspricht.
        :param node: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        to_be_removed = [node]
        while to_be_removed:
            nd = to_be_removed.pop(0)
            if self.nodes[nd]:
                to_be_removed += self.nodes[nd]
            del self.nodes[nd]

    def contains(self, node: tuple):
        """
        Prueft, ob der Knoten node im Dimensionsbaum vorhanden ist.
        :param node: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        return node in self.nodes

    def contained_in_right_subtree_from(self, root: tuple, node: tuple):
        """
        Prueft, ob der Knoten node im rechten Subtree der Wurzel root enthalten ist. root kann dabei ein beliebiger
        Knoten des Dimensionsbaums sein und muss nicht zwingend der tatsaechlichen Wurzel entsprechen.
        :param root: Ein tuple mit integer Eintraegen
        :param node: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        if not isinstance(root, tuple):
            raise TypeError("Argument 'root' = {}: {} ist kein tuple.".format(root, type(root)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if root not in self.get_nodes():
            raise ValueError("Argument 'root' = {}: Es existiert kein entsprechender Knoten.".format(root))

        return not self.contained_in_left_subtree_from(root, node)

    def contained_in_left_subtree_from(self, root: tuple, node: tuple):
        """
        Prueft, ob der Knoten node im linken Subtree der Wurzel root enthalten ist. root kann dabei ein beliebiger
        Knoten des Dimensionsbaums sein und muss nicht zwingend der tatsaechlichen Wurzel entsprechen.
        :param root: Ein tuple mit integer Eintraegen
        :param node: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        if not isinstance(root, tuple):
            raise TypeError("Argument 'root' = {}: {} ist kein tuple.".format(root, type(root)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if root not in self.get_nodes():
            raise ValueError("Argument 'root' = {}: Es existiert kein entsprechender Knoten.".format(root))

        children_of_root = self.get_children(root)
        if children_of_root:
            return self.contained_in_subtree_from(children_of_root[0], node)
        else:
            # root hat keine Kinder, entsprechend kann node auch nicht im linken Subtree von root liegen
            return False

    def contained_in_subtree_from(self, root: tuple, node: tuple):
        """
        Prueft, ob der Knoten node im Subtree der Wurzel root enthalten ist. root kann dabei ein beliebiger
        Knoten des Dimensionsbaums sein und muss nicht zwingend der tatsaechlichen Wurzel entsprechen.
        :param root: Ein tuple mit integer Eintraegen
        :param node: Ein tuple mit integer Eintraegen
        """
        # Typechecks
        if not isinstance(node, tuple):
            raise TypeError("Argument 'node' = {}: {} ist kein tuple.".format(node, type(node)))
        if not isinstance(root, tuple):
            raise TypeError("Argument 'root' = {}: {} ist kein tuple.".format(root, type(root)))
        # Valuechecks
        if node not in self.get_nodes():
            raise ValueError("Argument 'node' = {}: Es existiert kein entsprechender Knoten.".format(node))
        if root not in self.get_nodes():
            raise ValueError("Argument 'root' = {}: Es existiert kein entsprechender Knoten.".format(root))

        return self.get_subtree(root).contains(node)

    def copy(self):
        """
        Gibt eine Kopie des Dimensionsbaumobjektes zurueck.
        :return: dimtree Objekt
        """
        return deepcopy(self)

