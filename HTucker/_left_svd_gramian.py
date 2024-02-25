import torch


def left_svd_gramian(x):
    """
    Unter der Annahme, dass x eine reduzierte Gram'sche Matrix der Form x = v.T @ v ist, werden die linken
    Singulaervektoren und Singulaerwerte in absteigender Reihenfolge zurueckgegeben.
    :param x: torch.tensor
    :return: 2D-torch.tensor, 1D-torch.tensor
    """
    # Spektralzerlegung
    eig_val, Q = torch.linalg.eigh(x)
    # Singulaerwerte entsprechen den Quadratwurzeln der Eigenwertbetraege
    s_val = torch.sqrt(torch.abs(eig_val))
    # Index zur absteigenden Sortierung der Singulaerwerte
    desc_idc = torch.argsort(s_val, dim=0, descending=True)
    return Q[:, desc_idc], s_val[desc_idc]
