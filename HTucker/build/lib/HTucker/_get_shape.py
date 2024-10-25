

def get_shape(self):
    """
    Gibt die Shape des hierarchischen Tuckertensors 'self' zurueck.
    Die shape entspricht der Dimensionsgroessen des hierarchischen Tuckertensors.
    ______________________________________________________________________
    Output:
    (tuple,): Das mit tuple mit integer Eintraegen entspricht der shape von 'self'.
    ______________________________________________________________________
    Beispiel:
    a)
    x = HTTensor.randn((3,4,5,6))
    x.get_shape()    # = (3,4,5,6)

    b)
    x = HTTensor.randn((10,12))
    x.get_shape()    # = (10,12)
    """

    shape = []
    for dim in sorted(self.U.keys()):
        shape += [self.U[dim].shape[0]]
    return tuple(shape)