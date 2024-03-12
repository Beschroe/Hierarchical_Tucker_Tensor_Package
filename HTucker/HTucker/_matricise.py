import torch
from math import prod


def matricise(x: torch.Tensor, t: tuple):
    """
    Berechnet die t-Matrizierung des torch.Tensors 'x'. Das Tupel t enthaelt hierbei jene Dimensionen, die in der zu
    berechnenden Matrizierung als Zeilenindizes zusammengefasst werden.
    ______________________________________________________________________
    Parameter:
    x torch.Tensor: Der zu matrizierende torch.Tensor
    t (int,...): Die Dimensionen, die in der Matrizierung die Zeilenindizes formen
    ______________________________________________________________________
    Output:
    (2D torch.Tensor,): Die t-Matrizierung
    ______________________________________________________________________
    Beispiel:
    x = torch.randn(3,4,5,6), x.shape -> (3,4,5,6)
    xmat1 = matricise(x, (0,1))    # xmat1.shape -> (12,30)
    xmat2 = matricise(x, (1,))     # xmat2.shape -> (4,90)
    xmat3 = matricise(x, (1,3))    #xmat3.shape -> (24,15)
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


def dematricise(A: torch.Tensor, shape: tuple, t: tuple):
    """
    Berechnet den urspruenglichen Tensor einer Matrizierung.
    x = dematricise(A, shape, t) entspricht der inversen Operation von shape = x.shape, A = matricise(x, t).
    ______________________________________________________________________
    Parameter:
    - A 2D torch.Tensor: Entspricht einer t-Matrizierung
    - shape (int,...): Die urspruengliche shape des noch nicht matrizierten Tensors
    - t (int,...): Die Dimensionen des urspruenglichen Tensors, die gemeinsam in A die Zeilenindizes formen
    ______________________________________________________________________
    Output:
    (torch.Tensor,): Der dematrizierte Tensor.
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