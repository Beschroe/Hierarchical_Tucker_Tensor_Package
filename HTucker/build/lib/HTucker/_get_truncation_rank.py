import torch


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
    (int,): Der berechnete hierarchische Rang
    ______________________________________________________________________
    """
    if not isinstance(sv, torch.Tensor):
        raise TypeError("Argument 'sv': type(sv)={} | sv ist kein np.ndarray.".format(type(sv)))
    if len(sv.shape) != 1:
        raise TypeError("Argument 'sv': sv ist ein {}D-np.ndarray,"
                        " waehrend ein 1D-np.ndarray gefordert ist.".format(len(sv.shape)))
    cls._check_opts(opts)
    # Werden die ersten k Singulaervektoren mitgenommen, ist der Fehler in Frobeniusnurm durch sv_sum[k] gegeben
    sv_sum = torch.hstack((torch.sqrt(torch.cumsum((sv ** 2).flip(0), 0)).flip(0), torch.zeros(1)))
    rank_err_tol_abs = 1
    rank_err_tol_rel = 1
    if "err_tol_abs" in opts.keys():
        rank_err_tol_abs = max(int((sv_sum < opts["err_tol_abs"]).nonzero()[0]), 1)
    if "err_tol_rel" in opts.keys():
        rank_err_tol_rel = max(int((sv_sum < opts["err_tol_rel"] * torch.linalg.norm(sv)).nonzero()[0]), 1)
    truncation_rank = max(rank_err_tol_abs, rank_err_tol_rel)
    if "max_rank" in opts.keys():
        truncation_rank = min(truncation_rank, opts["max_rank"])
    return truncation_rank

