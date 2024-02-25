import torch
from dimtree.dimtree import dimtree
def check_U(U):
        """
        Prueft, ob U fuer sich genommen ein gueltiges Blattmatrixdict darstellt. Ist dies nicht der Fall, wird eine
        entsprechende Fehlermeldung geworfen
        :param U: dict: tuple:int -> torch.tesnor
        """
        if not isinstance(U, dict):
            raise TypeError("Argument 'U': type(U)={} | U ist kein dict.".format(type(U)))
        for k, v in U.items():
            # Checks von k
            if not isinstance(k, tuple):
                raise TypeError("Argument 'U': Der key {} ist nicht vom Typ tuple.".format(k))
            if len(k) != 1:
                raise ValueError("Argument 'U': Der key {} ist kein 1-tuple.".format(k))
            if not isinstance(k[0], int):
                raise TypeError("Argument 'U': Der Eintrag des keys {} ist kein int.".format(k))
            if k[0] < 0:
                raise ValueError("Argument 'U': Der Eintrag des keys {} ist kein nicht-negativer int.".format(k))
            # Checks von v
            if not isinstance(v, torch.Tensor):
                raise TypeError("Argument 'U': Der value des keys {} ist vom Typ {} und"
                                " damit kein torch.tesnor.".format(k, type(v)))
            if len(v.shape) != 2:
                raise ValueError("Argument 'U': Der value des keys {} ist ein {}D-torch.tesnor, muesste aber"
                                 " ein 2D-torch.tesnor sein.".format(k, len(v.shape)))
        if sorted([key[0] for key in U.keys()]) != list(range(max(U.keys())[0] + 1)):
            raise ValueError("Argument 'U': Die keys {} sind ungueltig, da nicht alle Dimensionen zwischen"
                             "0 und {} als keys auftreten.".format(U.keys(), max(U.keys())[0]))
        
def check_B(B):
        """
        Prueft, ob B fuer sich genommen ein gueltiges Transfertensordict darstellt. Ist dies nicht der Fall, wird eine
        entsprechende Fehlermeldung geworfen.
        :param B: dict: tuple:int -> torch.Tensor
        """
        if not isinstance(B, dict):
            raise TypeError("Argument 'B': type(B)={} | B ist kein dict.".format(type(B)))
        # Einzelne Pruefung aller key:value Paare
        for k, v in B.items():
            # Pruefung des keys
            if not isinstance(k, tuple):
                raise TypeError("Argument 'B': Der key {} ist nicht vom Typ tuple.".format(k))
            if not all(isinstance(item, int) for item in k):
                raise TypeError("Argument 'B': Der key {} enthaelt nicht-integer Eintraege.".format(k))
            if any(item < 0 for item in k):
                raise ValueError("Argument 'B': Der key {} enthaelt negative integer Eintraege.".format(k))
            if len(k) < 2:
                raise ValueError("Argument 'B': Der key {} repraesentiert keinen Dimensionscluster. Damit ist ein"
                                 " Transfertensor nicht angemessen.".format(k))
            # Pruefung des values
            if not isinstance(v, torch.Tensor):
                raise TypeError("Argument 'B': Der value des keys {} ist vom Typ {}"
                                " und damit kein torch.tensor".format(k, type(v)))
            # B ist ein 3D torch.tensor
            if not len(v.shape) == 3:
                raise ValueError("Argument 'B': Der value des keys {} ist"
                                 " ein {}D-torch.tensor und kein 3D-torch.tensor".format(k, len(v.shape)))
        # Pruefe, ob die Keys des dicts gueltige Dimensionscluster sind
        dims = set(range(max(len(key) for key in B.keys())))
        for k in B.keys():
            # Pruefe, ob k als geordnete sublist in dims enthalten ist
            contained = False
            if set(k) <= dims:
                contained = True
            if not contained:
                raise ValueError("Argument 'B': Der key {} repraesentiert keinen"
                                 " gueltigen Dimensionscluster.".format(k))

            
def check_compatibility(cls, U, B, dtree):
        """
        Prueft, ob das Blattmatrixdict U und das Transfertensordict B unter der in dtree hinterlegten
        Dimensionshierarchie kompatibel sind. Ist dies nicht der Fall, wird eine entsprechende Fehlermeldung geworfen.
        :param U: dict: tuple:int -> torch.Tensor
        :param B: dict: tuple:int -> torch.Tensor
        :param dtree: dt.dimtree
        """
        if not isinstance(U, dict):
            raise TypeError("Argument 'U': type(U)={} | U ist kein dict.".format(type(U)))
        if not isinstance(B, dict):
            raise TypeError("Argument 'B': type(B)={} | B ist kein dict.".format(type(B)))
        if not isinstance(dtree, dimtree):
            raise TypeError("Argument 'dtree': type(dtree)={} |"
                            " dtree ist nicht vom Typ ht.dimtree.".format(type(dtree)))

        # Alle Knoten aus dtree sind in U und B als keys enthalten
        if set(dtree.get_nodes()) != set(list(U.keys()) + list(B.keys())):
            raise ValueError("Argument 'U', 'B', 'dtree': U, B und dtree sind nicht kompatibel.")
        # U und B teilen sich keine keys
        if len(set(U.keys()).intersection(set(B.keys()))) > 0:
            raise ValueError("Argument 'U', 'B': U und B sind wegen geteilter keys nicht kompatibel.")
        # Die Raenge der Matrizen in U passen zu den Transfertensorgroessen in B
        for key in [key for key in B.keys() if len(key) == 2]:
            left = (key[0],)
            right = (key[1],)
            if left not in U.keys():
                raise ValueError("Argument 'U', 'B': U und B sind nicht kompatibel. B enthält den key {}, waehrend"
                                 "in U ({},) nicht als key vorhanden ist.".format(key, key[0]))
            if right not in U.keys():
                raise ValueError("Argument 'U', 'B': U und B sind nicht kompatibel. B enthält den key {}, waehrend"
                                 "in U ({},) nicht als key vorhanden ist.".format(key, key[1]))
            if U[left].shape[1] != B[key].shape[0] or U[right].shape[1] != B[key].shape[1]:
                raise ValueError("Argument 'U', 'B': U und B sind nicht kompatibel. B[{}].shape={} und"
                                 " U[{}].shape={} sowie U[{}].shape={}.".format(key, B[key].shape, left, U[left].shape,
                                                                               right, U[right].shape))
        # Die Raenge der Transfertensoren in B passen zu der Eltern-Kind-Relation in dtree
        for key in [key for key in B.keys() if len(key) > 2]:
            if key not in dtree.get_inner_nodes():
                raise ValueError("Argument 'B', 'dtree': B und dtree sind nicht konsistent. B enthaelt einen Knoten"
                                 " als key, der in dtree nicht als innerer Knoten auftritt.")
            left = dtree.get_left(key)
            right = dtree.get_right(key)
            if left+right != key:
                raise ValueError("Argument 'dtree': Der Dimensionsbaum ist defekt. Die Kinder des Knotens {} "
                                 "lauten {} und {} und repraesentieren damit nicht die Dimensionen des"
                                 " Elternknotens.".format(key, left, right))
            if dtree.is_leaf(left):
                if left not in U.keys():
                    raise ValueError("Argument 'U', 'dtree': Fuer den Knoten {} definiert dtree das linke Kind als {}."
                                     " Dieses findet sich allerdings nicht in U.".format(key, left))
                if U[left].shape[1] != B[key].shape[0]:
                    raise ValueError("Argument 'B', 'U', 'dtree': Die shape des Transfertensors des Elternknoten {} lautet"
                                     " {} und ist damit nicht"
                                     " kompatibel zur shape der Blattmatrix des linken Kindes"
                                     " {} mit {}.".format(key, B[key].shape, left, U[left].shape))
            else:
                if left not in B.keys():
                    raise ValueError("Argument 'B', 'dtree': Fuer den Knoten {} definiert dtree das linke Kind als {}."
                                     " Dieses findet sich allerdings nicht in B.".format(key, left))
                if B[left].shape[2] != B[key].shape[0]:
                    raise ValueError("Argument 'B', 'dtree': Die shape des Transfertensors des Elternknoten {} lautet"
                                     " {} und ist damit nicht"
                                     " kompatibel zur shape des Transfertensors des linken Kindes"
                                     " {} mit {}.".format(key, B[key].shape, left, B[left].shape))
            if dtree.is_leaf(right):
                if right not in U.keys():
                    raise ValueError("Argument 'U', 'dtree': Fuer den Knoten {} definiert dtree das rechte Kind als {}."
                                     " Dieses findet sich allerdings nicht in U.".format(key, right))
                if U[right].shape[1] != B[key].shape[1]:
                    raise ValueError("Argument 'B', 'dtree', 'U': Die shape des Transfertensors des Elternknoten {} "
                                     "lautet {} und ist damit nicht"
                                     " kompatibel zur shape des Transfertensors des rechten Kindes"
                                     " {} mit {}.".format(key, B[key].shape, right, U[right].shape))
            else:
                if right not in B.keys():
                    raise ValueError("Argument 'B', 'dtree': Fuer den Knoten {} definiert dtree das rechte Kind als {}."
                                     " Dieses findet sich allerdings nicht in B.".format(key, right))
                if B[right].shape[2] != B[key].shape[1]:
                    raise ValueError("Argument 'B', 'dtree': Die shape des Transfertensors des Elternknoten {} lautet"
                                     " {} und ist damit nicht"
                                     " kompatibel zur shape des Transfertensors des rechten Kindes"
                                     " {} mit {}.".format(key, B[key].shape, right, B[right].shape))


def check_opts(opts):
        """
        Prueft, ob das dict opts folgende Kriterien erfuellt:
            - Mindestens einer der Keys 'max_rank', 'err_tol_abs' oder 'err_tol_rel' ist enthalten
            - Der Value zu 'max_rank' muss ein positiver integer sein
            - Der Value zu 'err_tol_abs' muss ein positiver float sein
            - Der Value zu 'err_tol_abs' muss ein positiver float sein
        :param opts: dict
        """
        if not isinstance(opts, dict):
            raise TypeError("Argument 'opts': type(opts)={} | opts ist kein dict.".format(type(opts)))
        for k, v in opts.items():
            if not isinstance(k, str):
                raise TypeError("Argument 'opts': opts enthaelt einen ungueltigen key. type({})={}.".format(k, type(k)))
            if k not in {'max_rank', 'err_tol_abs', 'err_tol_rel'}:
                raise ValueError("Argument 'opts': der key {} in opts"
                                 " ist nicht erlaubt. Erlaubt sind: {}.".format(k, {'max_rank', 'err_tol_abs',
                                                                                    'err_tol_rel'}))
            if k == "max_rank":
                if not isinstance(v, int):
                    raise TypeError("Argument 'opts': Der value des keys {}"
                                    " ist kein integer. type(value)={}.".format(k, type(v)))
                if v < 1:
                    raise ValueError("Argument 'opts': Der value {} des keys {}"
                                     " ist kein positiver integer.".format(v, k))
            else:
                # k in ['err_tol_abs', 'err_tol_rel']
                if not isinstance(v, float):
                    raise TypeError("Argument 'opts': Der value des keys {} ist"
                                    " kein float. type(value)={}.".format(k, type(v)))
                if v <= 0.0:
                    raise ValueError("Argument 'opts': Der value {} des keys {} ist kein positiver float.".format(v, k))