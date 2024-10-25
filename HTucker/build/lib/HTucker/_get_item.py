from copy import deepcopy


def get(self, key: int | slice | tuple):
    """
    Gibt den durch key bestimmten Eintrag des hierarchischen Tuckertensors 'self' zurueck.
    Das Slicing Verhalten ist hierbei analog zu dem von pytorch.
    ______________________________________________________________________
    Parameter:
    - key int | slice | tuple: Der Index bzw. Slice
    ______________________________________________________________________
    Output:
    (float | HTucker.HTTensor, ): Das referenzierte Element bzw. der geslicete hierarchische Tuckertensor
    ______________________________________________________________________
    Beispiel:
                 HTucker.HTTensor              <~~~>            torch.Tensor
    a) int
       x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
       x = x[0]                                  |           x = x[0]
       type(x)    # HTucker.HTTensor             |           type(x)    # torch.Tensor
       x.get_shape()    # (4,5,6)                |           x.shape    # torch.size([4,5,6])


    a) slice
       x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
       x = x[0,:,1,:]                            |           x = x[0,:,1,:]
       type(x)    # HTucker.HTTensor             |           type(x)    # torch.Tensor
       x.get_shape()    # (4,6)                  |           x.shape    # torch.size([4,6])

    a) tuple
       x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
       x = x[2,1,0,4]                            |           x = x[2,1,0,4]
       type(x)    # float                        |           type(x)    # torch.Tensor
                                                 |           x.shape    # torch.size([])
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

