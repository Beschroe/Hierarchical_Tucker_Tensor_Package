import torch


def left_svd_gramian(x: torch.Tensor):
    """
    Berechnet die linken Singulaervektoren samt Singulaerwerte einer Matrix
    v ueber ihre Gram-Matrix x=v@v.T
    ______________________________________________________________________
    Parameter:
    - x 2D torch.Tensor
    ______________________________________________________________________
    Output:
    (2D torch.Tensor, 1D torch.Tensor): Der erste Eintrag entspricht den linken Singulaervektoren, waehred der zweite
                                        Eintrag den zugehoerigen Singulaerwerten entspricht. Das Tupel ist bezogen
                                        auf die Singulaerwerte in absteigender Reihenfolge sortiert.
    """


    ## Spektralzerlegung
    eig_val, Q = torch.linalg.eigh(x)
    # Singulaerwerte entsprechen den Quadratwurzeln der Eigenwertbetraege
    s_val = torch.sqrt(torch.abs(eig_val))
    # Index zur absteigenden Sortierung der Singulaerwerte
    desc_idc = torch.argsort(s_val, dim=0, descending=True)
    return Q[:, desc_idc], s_val[desc_idc]

