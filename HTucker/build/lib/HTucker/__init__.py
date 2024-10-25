import torch


class HTTensor:
    # Importierte Instanzmethoden
    from ._full import full
    from ._get_shape import get_shape
    from ._nbytes import nbytes
    from ._get_item import get
    from ._get_order import get_order
    from ._squeeze import squeeze
    from ._scalar_mul import scalar_mul
    from ._mode_mul import mode_mul
    from ._orthogonalize import orthogonalize
    from ._get_gramians import _get_gramians
    from ._truncate_htt import truncate_htt
    from ._get_rank import get_rank
    from ._plus import plus
    from ._ele_mul import ele_mul
    from ._ele_mode_mul import ele_mode_mul
    from ._tensordot import tensordot
    from ._change_root import _change_root
    from ._minus import minus

    # Importierte Klassenmethoden
    from ._truncate import truncate
    from ._get_truncation_rank import _get_truncation_rank
    from ._truncate_sum import truncate_sum
    from ._get_gramians_sum import _get_gramians_sum
    from ._randn import randn
    truncate = classmethod(truncate)
    _get_truncation_rank = classmethod(_get_truncation_rank)
    truncate_sum = classmethod(truncate_sum)
    _get_gramians_sum = classmethod(_get_gramians_sum)
    randn = classmethod(randn)

    # Importierte statische Methoden
    from ._checks import _check_U, _check_B, _check_opts, _check_compatibility
    from ._matricise import matricise, dematricise
    from ._left_svd_gramian import left_svd_gramian
    from ._left_svd_qr import left_svd_qr
    _check_U = staticmethod(_check_U)
    _check_B = staticmethod(_check_B)
    _check_opts = staticmethod(_check_opts)
    _check_compatibility = staticmethod(_check_compatibility)
    matricise = staticmethod(matricise)
    dematricise = staticmethod(dematricise)
    left_svd_gramian = staticmethod(left_svd_gramian)
    left_svd_qr = staticmethod(left_svd_qr)

    def __init__(self, U, B, dtree, is_orthog=False):
        """
        Konstruktor
        Erzeugt aus dem Blattmatrixdict U und Transfertensordict B einen hierarchischen Tuckertensor, dessen Dimensions-
        hierarchie durch dtree vorgegeben ist. Der Parameter is_orthog zeigt hierbei an, ob der zu erzeugende hierarch-
        ische Tuckertensor orthogonal sein wird.
        :param U: dict: tuple:integer -> torch.Tensor
        :param B: dict: tuple:integer -> torch.Tensor
        :param dtree: dt.dimtree
        :param is_orthog: bool
        """
        # Argumentchecks: U
        HTTensor._check_U(U)
        # Argumentchecks: B
        HTTensor._check_B(B)
        # Argumentchecks: Kompatibilitaet
        HTTensor._check_compatibility(U, B, dtree)
        # Setzen der Instanzvariablen
        self.U = U
        self.B = B
        self.dtree = dtree
        self.is_orthog = is_orthog

    def __getitem__(self, key):
        return self.get(key)

    def __add__(self, y):
        return self.plus(y)

    def __sub__(self, y):
        return self.minus(y)

    def __mul__(self, y):
        if isinstance(y, int):
            return self.scalar_mul(float(y))
        elif isinstance(y, torch.Tensor):
            squeezed = y.squeeze()
            if len(squeezed.shape) == 0:
                return self.scalar_mul(float(squeezed))
            else:
                raise ValueError("Argument 'y': type(y)={} | y mit shape {} repraesentiert"
                                 " keinen Skalar.".format(type(y), y.shape))
        elif isinstance(y, float):
            return self.scalar_mul(y)
        else:
            raise TypeError("Argument 'y': type(y)={} | y muss vom Type float, int"
                            " oder torch.Tensor sein.".format(type(y)))

    def __imod__(self, opts):
        self._check_opts(opts)
        return self.truncate_htt(opts)

    def __mod__(self, opts):
        self._check_opts(opts)
        return self.truncate_htt(opts)