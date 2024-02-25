import torch

def left_svd_qr(x):
    """
    Berechnet die linken Singulaervektoren samt zugehoeriger Singulaerwerte der Matrix x.
    :param x: torch.Tensor
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