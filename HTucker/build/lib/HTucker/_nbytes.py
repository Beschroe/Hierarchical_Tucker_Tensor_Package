

def nbytes(self):
    """
    Gibt den wesentlichen Speicherbedarf des hierarchischen Tuckertensor 'self' zurueck. Dieser speist sich aus dem
    eigenommenen Speicherplatz aller Blattmatrizen und Transfertensoren.
    ______________________________________________________________________
    Output:
    (int,): Der eingenommene Speicherplatz in Bytes.
    ______________________________________________________________________
    Beispiel:
    a)
    x = torch.randn((3,4,5,6))
    xh = HTTensor.truncate(x)
    x.nbytes    # = 5872
    b) (Die Byteanzahl an diesem Beispiel variiert, da HTucker.randn() einen zufaelligen hierarchischen Rang waehlt)
    x = torch.randn((3,4,5,6))
    opts = {"max_rank": 10, "err_tol_abs": 1.0}
    xh = HTTensor.truncate(x, opts)
    xh.nbytes    # = 4848

    (Die Byteanzahl ist in Beispiel a) groesser, da dort als in b) keine Rangkuerzung vorgenommen wird)
    """
    mem = 0
    for matrix in self.U.values():
        mem += matrix.numel() * matrix.element_size()
    for tensor in self.B.values():
        mem += tensor.numel() * tensor.element_size()
    return mem