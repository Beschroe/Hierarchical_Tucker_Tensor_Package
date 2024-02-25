import torch
from copy import deepcopy


def get(self, key):
    """
    Gibt das Element bzw. die Elemente, das/die durch key referenziert werden, zurueck.
    @param key: tuple, slice oder int
    @return: float oder HTucker.HTucker
    """
    if isinstance(key, int):
        return get_from_int(self, key)
    elif isinstance(key, slice):
        return get_from_slice(self, key)
    elif isinstance(key, tuple):
        return get_from_tuple(self, key)
    else:
        raise TypeError("Argument 'key': type(key)={} | key ist weder int, tuple noch slice.".format(type(key)))


def get_from_slice(self, key):
    if not isinstance(key, slice):
        raise TypeError("Argument 'key': type(key)={} | key ist kein slice Objekt.".format(type(key)))
    z = deepcopy(self)
    # Es werden nur die in key definierten Zeilen behalten
    z.U[(0,)] = z.U[(0,)][key, :]
    # Entfernen der Singletondimension, falls vorhanden
    if z.U[(0,)].shape[0] == 1:
        z = z.squeeze()
    return z


def get_from_int(self, key):
    if not isinstance(key, int):
        raise TypeError("Argument 'key': type(key)={} | key ist kein int.".format(type(key)))
    shape = self.get_shape()
    if key not in range(shape[0]):
        raise ValueError("Argument 'key': key={} ist kein gueltiger Index fuer eine Dimension der Groesse {}."
                         .format(key, shape[0]))
    z = deepcopy(self)
    # Es wird nur die key-te Zeile der Blattmatrix der 0-ten Dimension beibehalten
    z.U[(0,)] = z.U[(0,)][key, :].reshape(1, -1)
    # Entfernen der Singletondimension
    z = z.squeeze()
    return z


def get_from_tuple(self, key):
    if not isinstance(key, tuple):
        raise TypeError("Argument 'key': type(key)={} | key ist kein tuple.".format(type(key)))
    shape = self.get_shape()
    z = deepcopy(self)
    # Iteriere ueber Tupeleintraege
    for counter, idx in enumerate(key):
        if isinstance(idx, int):
            z.U[(counter,)] = z.U[(counter,)][idx, :].reshape(1,-1)
        elif isinstance(idx, slice):
            z.U[(counter,)] = z.U[(counter,)][idx, :]
        else:
            raise TypeError("Argument 'key': key enthaelt ungueltige Eintraege, die weder vom Typ int noch slice sind.")
    if any(mat.shape[0] == 1 for mat in z.U.values()):
        # Entferne Singleton Dimensionen falls vorhanden
        z = z.squeeze()
    return z

