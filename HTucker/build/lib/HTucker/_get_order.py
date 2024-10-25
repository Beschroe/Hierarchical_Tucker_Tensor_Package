

def get_order(self):
    """
    Gibt die Ordnung des hierarchischen Tuckertensors 'self' zurueck.
    ______________________________________________________________________
    Output:
    (int,): Die Ordnung von 'self'
    ______________________________________________________________________
    Beispiel:
    a)
    x = HTTensor.randn((3,4,5,6))
    x.get_order()    # = 4

    b)
    x = HTTensor.randn((3,4))
    x.get_order()    # = 2
    """
    return len(self.get_shape())
