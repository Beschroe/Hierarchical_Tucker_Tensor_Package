

def get_shape(self):
    """
    Gibt die Shape des hierarchischen Tuckertensors zurueck.
    :return: tuple:int
    """
    shape = []
    for dim in sorted(self.U.keys()):
        shape += [self.U[dim].shape[0]]
    return tuple(shape)