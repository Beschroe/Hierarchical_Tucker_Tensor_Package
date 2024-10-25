import torch

def left_svd_qr(x: torch.Tensor):
    """
    Berechnet die linken Singulaervektoren samt Singulaerwerte der Matrix 'x'.
    ______________________________________________________________________
    Parameter:
    x 2D torch.Tensor
    ______________________________________________________________________
    Output:
    (2D torch.Tensor, 1D torch.Tensor): Der erste Eintrag entspricht den linken Singulaervektoren, waehred der zweite
                                        Eintrag den zugehoerigen Singulaerwerten entspricht. Das Tupel ist bezogen
                                        auf die Singulaerwerte in absteigender Reihenfolge sortiert.

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Argument 'x': type(x)={} | x ist kein torch.Tensor.".format(type(x)))
    if len(x.shape) != 2:
        raise ValueError("Argument 'x': x.shape={} | x ist kein 2D-torch.Tensor.".format(x.shape))

    if x.shape[0] > x.shape[1]:
        q, r = torch.linalg.qr(x, mode="reduced")
        u, s, _ = torch.linalg.svd(r, full_matrices=False)
        u = q @ u
    else:
        _, r = torch.linalg.qr(x.T, mode="r")
        u, s, _ = torch.linalg.svd(r.T, full_matrices=False)
    return u, s