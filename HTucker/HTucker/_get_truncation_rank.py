import torch
from warnings import warn


def _get_truncation_rank(cls, sv: torch.Tensor, opts: dict):
    """
    Hinweis: Dies ist eine innere Funktion der Funktionen truncate, truncate_htt, ele_mul und trunc_sum.
    ______________________________________________________________________
    Berechnet auf Grundlage der in 'sv' enthaltenen Singulaerwerte den minimalen hierarchischen Rang, der die
    Constraints aus 'opts' einhaelt.
    ______________________________________________________________________
    Parameter:
    - sv 1D-torch.Tensor: Der 1D Tensor enthaelt absteigend sortierte Singulaerwerte.
    - opts dict: Das Optionen-dict kann folgende Constraints enthalten:
                                    - "max_rank": positiver integer | Legt den maximalen hierarchischen Rang
                                                  fest
                                    - "err_tol_abs": positiver float | Legt die einzuhaltende absolute
                                                     Fehlertoleranz fest
                                    - "err_tol_rel": positiver float | Left die einzuhaltende relative
                                                     Fehlertoleranz fest
    ______________________________________________________________________
    Output:
    1D-torch.Tensor: Der berechnete hierarchische Rang
    ______________________________________________________________________
    """
    if not isinstance(sv, torch.Tensor):
        raise TypeError("Argument 'sv': type(sv)={} | sv ist kein torch.Tensor.".format(type(sv)))
    if len(sv.shape) != 1:
        raise TypeError("Argument 'sv': sv ist ein {}D-torch.Tensor,"
                        " waehrend ein 1D-torch.Tensor gefordert ist.".format(len(sv.shape)))
    cls._check_opts(opts)
    atol = opts["err_tol_abs"]
    rtol = opts["err_tol_rel"]
    max_rank = opts["max_rank"]
    # compute cumsum
    cum_sv = torch.cat([torch.sqrt(torch.cumsum(sv.flip(dims=(0,))**2,dim=0)).flip(dims=(0,)), torch.zeros(1)])
    # compute rank_a
    if atol:
        rank_a = torch.max((cum_sv <= atol).nonzero()[0,0], torch.tensor(1))
    else:
        rank_a = torch.tensor(1)
    # compute rank_r
    if rtol:
        rank_r = torch.max((cum_sv <= rtol*torch.linalg.norm(sv.type(torch.float))).nonzero()[0,0], torch.tensor(1))
    else:
        rank_r = torch.tensor(1)
    rank = torch.max(rank_r, rank_a)
    if max_rank:
        if rank > max_rank:
            warn("Requested greater truncation rank than allowed -> Error boundary potentially broken.")
            rank = max_rank
    return rank

