from copy import deepcopy


def scalar_mul(self, c: float):
    """
    Berechnet das Produkt des hierarchischen Tuckertensors 'self' mit dem Skalar 'c'.
    Hinweis: Die Verwendung des Operators "*" ist moeglich. Dabei entspricht x.scalar_mul(c) dem Ausdruck x * c.
             Die Formulierung als c * x ist nicht moeglich. Der beteiligte hierarchische Tuckertensor muss stets
             der erste Operand sein.
    ______________________________________________________________________
    Parameter:
    - c float: Der Skalar mit dem 'self' multipliziert wird
    ______________________________________________________________________
    Output:
    (HTucker.HTTensor,): Das Produkt gegebn als hierarchischer Tuckertensor
    ______________________________________________________________________
    Beispiel:
                  HTucker.HTTensor             <~~~>            torch.Tensor
    a) x = HTTensor.randn((3,4,5,6))              |           x = torch.randn(3,4,5,6)
       c = 3.1415926                             |           c = 3.1415926
       prod = x.scalar_mul(c)                    |           prod = x * c

    a) Die Verwendung des Operators "*" ist
       moeglich.
       x = HTTensor.randn((3,4,5,6))              |           x = torch.randn(3,4,5,6)
       c = 2.718281828                           |           c = 2.718281828
       prod = x * c                              |           prod = x * c
    """

    if not isinstance(c, float):
        raise TypeError("Argument 'c': type(c)={} | c ist kein float".format(type(c)))

    # Erzeuge Kopie
    x = deepcopy(self)

    # Multipliziere den Transfertensor der Wurzel mit dem Skalar
    x.B[x.dtree.get_root()] = x.B[x.dtree.get_root()] * c
    # Setze is_orthog Flag auf False
    x.is_orthog = False
    return x
