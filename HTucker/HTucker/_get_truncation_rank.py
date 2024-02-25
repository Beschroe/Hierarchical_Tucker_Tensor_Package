import torch


def get_truncation_rank(cls, sv, opts):
    """
    Gibt basierend auf sv (absteigend sortierte Singulaerwerte) und opts den minimalen Rang zurueck, sodass die in
    opts hinterlegten constraints erfuellt werden. 'max_rank' wiegt dabei schwerer als die einzuhaltenden Fehler-
    schranken.
    :param sv: list: float
    :param opts: dict
    :return: int
    """
    if not isinstance(sv, torch.Tensor):
        raise TypeError("Argument 'sv': type(sv)={} | sv ist kein np.ndarray.".format(type(sv)))
    if len(sv.shape) != 1:
        raise TypeError("Argument 'sv': sv ist ein {}D-np.ndarray,"
                        " waehrend ein 1D-np.ndarray gefordert ist.".format(len(sv.shape)))
    cls.check_opts(opts)
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

