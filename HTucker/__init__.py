from .dimtree import dimtree


class HTucker:
    # Importierte Instanzmethoden
    from HTucker._full import full
    from HTucker._get_shape import get_shape
    from HTucker._get_size_in_bytes import get_size_in_bytes
    from HTucker._get_item import get
    from HTucker._get_order import get_order
    from HTucker._squeeze import squeeze
    from HTucker._scalar_mul import scalar_mul
    from HTucker._mode_mul import mode_mul
    from HTucker._orthogonalize import orthogonalize
    from HTucker._get_gramians import get_gramians
    from HTucker._truncate_htt import truncate_htt
    from HTucker._get_rank import get_rank
    from HTucker._plus import plus
    from HTucker._ele_mul import ele_mul
    from HTucker._ele_mode_mul import ele_mode_mul

    # Importierte Klassenmethoden
    from HTucker._truncate import truncate
    from HTucker._get_truncation_rank import get_truncation_rank
    from HTucker._truncate_sum import truncate_sum
    from HTucker._get_gramians_sum import get_gramians_sum
    from HTucker._checks import check_compatibility
    truncate = classmethod(truncate)
    get_truncation_rank = classmethod(get_truncation_rank)
    truncate_sum = classmethod(truncate_sum)
    get_gramians_sum = classmethod(get_gramians_sum)
    check_compatibility = classmethod(check_compatibility)

    # Importierte statische Methoden
    from HTucker._checks import check_U, check_B, check_opts
    from HTucker._matricise import matricise, dematricise
    from HTucker._left_svd_gramian import left_svd_gramian
    from HTucker._left_svd_qr import left_svd_qr
    check_U = staticmethod(check_U)
    check_B = staticmethod(check_B)
    check_opts = staticmethod(check_opts)
    matricise = staticmethod(matricise)
    dematricise = staticmethod(dematricise)
    left_svd_gramian = staticmethod(left_svd_gramian)
    left_svd_qr = staticmethod(left_svd_qr)

    def __init__(self, U, B, dtree, is_orthog=False):
        """
        Konstruktor
        Erzeugt aus dem Blattmatrixdict U und Transfertensordict B einen hierarchischen Tuckertensor dessen Dimensions-
        hierarchie durch dtree vorgegeben ist. Der Parameter is_orthog zeigt hierbei an, ob der zu erzeugende hierarch-
        ische Tuckertensor orthogonal sein wird.
        :param U: dict: tuple:integer -> torch.Tensor
        :param B: dict: tuple:integer -> torch.Tensor
        :param dtree: dt.dimtree
        :param is_orthog: bool
        """
        # Argumentchecks: U
        HTucker.check_U(U)
        # Argumentchecks: B
        HTucker.check_B(B)
        # Argumentchecks: Kompatibilitaet
        HTucker.check_compatibility(U, B, dtree)
        # Setzen der Instanzvariablen
        self.U = U
        self.B = B
        self.dtree = dtree
        self.is_orthog = is_orthog

    def __getitem__(self, key):
        return self.get(key)
