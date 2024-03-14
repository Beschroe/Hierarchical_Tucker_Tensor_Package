import torch
from scipy.linalg import svd


def left_svd_gramian(x: torch.Tensor):
    """
    Berechnet die linken Singulaervektoren samt Singulaerwerte der Matrix 'v' und gibt diese absteigend sortiert
    zurueck.
    Die Berechnungen beruhen auf der Annahme, dass 'v' eine reduzierte Gram'sche Matrix der Form v= x.T @ x ist.
    ______________________________________________________________________
    Parameter:
    - x 2D torch.Tensor
    ______________________________________________________________________
    Output:
    (2D torch.Tensor, 1D torch.Tensor): Der erste Eintrag entspricht den linken Singulaervektoren, waehred der zweite
                                        Eintrag den zugehoerigen Singulaerwerten entspricht. Das Tupel ist bezogen
                                        auf die Singulaerwerte in absteigender Reihenfolge sortiert.
    """

    if x.is_cuda:
        ## Spektralzerlegung
        eig_val, Q = torch.linalg.eigh(x)
        # Singulaerwerte entsprechen den Quadratwurzeln der Eigenwertbetraege
        s_val = torch.sqrt(torch.abs(eig_val))
        # Index zur absteigenden Sortierung der Singulaerwerte
        desc_idc = torch.argsort(s_val, dim=0, descending=True)
        return Q[:, desc_idc], s_val[desc_idc]
    else:
        xnpy = x.numpy()
        Q, eig_val, _ = svd(xnpy, full_matrices=False, lapack_driver="gesvd")
    return torch.Tensor(Q), torch.sqrt(torch.Tensor(eig_val))
