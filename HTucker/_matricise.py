import torch
from math import prod


def matricise(x, t):
    """
    Berechnet die t-Matrizierung eines Tensors x. Die Reihenfolge der in t enthaltenen Dimensionen beeinflusst
    dabei das Resultat.
    Hinweis: Falls moeglich, wird ein View Objekt zurueckgegeben. Ist dies nicht moeglich, wird ein neuer torch.Tensor
             erzeugt. Fuer mehr Informationen: siehe torch.reshape.
    :param x: torch.Tensor
    :param t: tuple:int
    :return: torch.Tensor
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Argument 'x': type(x)={} | x ist kein torch.Tensor.".format(type(x)))
    if not isinstance(t, tuple):
        raise TypeError("Argument 't': type(t)={} | t ist kein tuple.".format(type(t)))
    # Sortiere die Modi von x um, sodass die in t enthaltenen Modi fuehrend sind
    x_moved_axis = torch.movedim(x, source=t, destination=tuple(range(len(t))))
    # Berechne die Anzahl an Zeilenindizes der t-Matrizierung
    nr_rows = prod(x.size()[mode] for mode in t)
    # Ordne x_moved_axis in entsprechende t-Matrizierung um
    x_matricised = x_moved_axis.reshape(nr_rows, -1)
    return x_matricised


def dematricise(A, shape, t):
    """
    Berechnet den vollen Tensor einer Matrizierung. x = dematricise(A, shape, t) entspricht der
    inversen Operation von A = matricise(x, t)
    Hinweis: Falls moeglich, wird ein View Objekt zurueckgegeben. Ist dies nicht moeglich, wird ein neuer torch.Tensor
         erzeugt. Fuer mehr Informationen: siehe torch.reshape.
    :param A: 2D torch.Tensor
    :param shape: tuple
    :param t: tuple
    :return: ND torch.Tensor
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("Argument 'A': type(A)={} | A ist kein torch.Tensor.".format(type(A)))
    if A.dim() != 2:
        raise ValueError("Argument 'A': A.dim()={} | A ist kein 2D torch.Tensor.".format(A.dim()))
    if not isinstance(shape, tuple):
        raise TypeError("Argument 'shape': type(shape)={} | shape ist kein tuple.".format(type(shape)))
    if not all(isinstance(item, int) for item in shape):
        raise TypeError("Argument 'shape': shape enthaelt nicht-integer Elemente.")
    if not all(item > 0 for item in shape):
        raise ValueError("Argument 'shape': shape enthaelt nicht-positive integer Elemente.")
    if not isinstance(t, tuple):
        raise TypeError("Argument 't': type(t)={} | t ist kein tuple.".format(type(t)))
    if not all(isinstance(item, int) for item in t):
        raise TypeError("Argument 't': t enthaelt nicht-integer Elemente.")
    if not all(item >= 0 for item in t):
        raise ValueError("Argument 't': t enthaelt nicht-positive integer Elemente.")
    if len(t) < 1:
        raise ValueError("Argument 't': t muss mindestens eine Dimension enthalten.")
    if len(set(t)) != len(t):
        raise ValueError("Argument 't': t enthaelt Duplikate.")
    if len(set(t)) != len(set(t).intersection(set(range(len(shape))))):
        raise ValueError("Argument 'shape' und 't': shape und t sind nicht kompatibel.")
    # Berechne die shape des urspruenglichen Tensors, nachdem dessen Modi mittels torch.movedim permutiert wurden
    col_dims = (dim for dim in range(len(shape)) if dim not in t)
    sh = tuple(shape[dim] for dim in t) + tuple(shape[dim] for dim in col_dims)
    # Reshape die Matrizierung in diese Form
    x = A.reshape(*sh)
    # Permutiere die Modi in die urspruengliche Reihenfolge
    x = torch.movedim(x, source=tuple(range(len(t))), destination=t)
    return x