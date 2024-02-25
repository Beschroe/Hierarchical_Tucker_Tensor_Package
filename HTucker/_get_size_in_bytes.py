

def get_size_in_bytes(self):
    """
    Gibt den Speicherbedarf der Tensoren in U und B in Bytes zurueck. Dieser Speicherbedarf entspricht im Wesentlichem
    dem Speicherbedarf des gesamten hierarchischen Tuckertensors.
    """
    mem = 0
    for matrix in self.U.values():
        mem += matrix.numel() * matrix.element_size()
    for tensor in self.B.values():
        mem += tensor.numel() * tensor.element_size()
    return mem