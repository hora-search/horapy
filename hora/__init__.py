import numpy
from .hora import BruteForceIndexUsize as HoraBruteForceIndexUsize
from .hora import BruteForceIndexStr as HoraBruteForceIndexStr
# from .hora import BPTIndex as HoraBPTIndex
from .hora import HNSWIndexUsize as HoraHNSWIndexUsize
from .hora import HNSWIndexStr as HoraHNSWIndexStr
from .hora import PQIndexUsize as HoraPQIndexUsize
from .hora import PQIndexStr as HoraPQIndexStr
from .hora import SSGIndexUsize as HoraSSGIndexUsize
from .hora import SSGIndexStr as HoraSSGIndexStr
from .hora import IVFPQIndexUsize as HoraIVFPQIndexUsize
from .hora import IVFPQIndexStr as HoraIVFPQIndexStr


class HoraANNIndex():
    """Hora Index Base Class

    Attributes:
        dimension: your index dimension
        dtype: the index type, only two option `usize` and `str`
    """

    def __init__(self, dimension, dtype):
        self.ann_idx = None
        self.dtype = dtype
        self.ann_type = None
        self.dimension = dimension
        self._accept_dtype = ["usize", "str"]
        self._args = dict()

    def __str__(self):
        return "Hora ANNIndex <{}> ({})".format(self.name(), self._format_params())

    def __repr__(self):
        return "Hora ANNIndex <{}> ({})".format(self.name(), self._format_params())

    def build(self, metrics=""):
        self.ann_idx.build(metrics)

    def add(self, vs, idx=None):
        """ add vector<float>

        Args:
            vs: list[float] or np.array(float32)
            idx: dtype
        """
        if isinstance(vs, list):
            self.ann_idx.add(vs, idx)
        elif isinstance(vs, numpy.ndarray):
            self.ann_idx.add(vs.tolist(), idx)
        else:
            raise TypeError(
                "invalid type {}, only accept list[float] and numpy.ndarray".format(type(vs)))

    def search(self, vs, k):
        """ search k nearest neighbors

        Args:
            vs: list[float] or np.array(float32)
        """
        if isinstance(vs, numpy.ndarray):
            return self.ann_idx.search_np(vs.astype("float32"), k)
        elif isinstance(vs, list):
            return self.ann_idx.search(vs, k)
        else:
            raise TypeError(
                "invalid type {}, only accept list[float] and numpy.ndarray".format(type(vs)))

    def name(self):
        """ index name
        """
        return self.ann_idx.name()

    def load(self, path):
        """ load index binary

        Args:
            path: file path
        """
        self.ann_idx = self.ann_type.load(path)

    def dump(self, path):
        """dump index into binary file

        Args:
            path: dump file path
        """
        self.ann_idx.dump(path)

    def _format_params(self):
        return ", ".join(["{}: {}".format(key, self._args[key]) for key in self._args])


class BruteForceIndex(HoraANNIndex):
    def __init__(self, dimension, dtype):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraBruteForceIndexUsize
        elif dtype == "str":
            self.ann_type = HoraBruteForceIndexStr
        else:
            raise TypeError("invalid dtype {}, only accept {}".format(
                dtype, self._accept_dtype))
        self.ann_idx = self.ann_type(dimension)
        self._args = {
            "dimension": dimension,
            "dtype": dtype
        }


# class BPTIndex(ANNIndex):
#     def __init__(self, dimension, tree_num, candidate_size, dtype):
#         super().__init__(dimension, dtype)
#         self.ann_type = HoraBPTIndex
#         self.ann_idx = HoraBPTIndex(dimension, tree_num, candidate_size)


class HNSWIndex(HoraANNIndex):
    """HNSWIndex

    the implementation of algorithm https://arxiv.org/abs/1603.09320
    """

    def __init__(self, dimension, dtype, max_item=1000000, n_neigh=32, n_neigh0=64, ef_build=20, ef_search=500, has_deletion=False):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraHNSWIndexUsize
        elif dtype == "str":
            self.ann_type = HoraHNSWIndexStr
        else:
            raise TypeError("invalid dtype {}, only accept {}".format(
                dtype, self._accept_dtype))

        self.ann_idx = self.ann_type(
            dimension, max_item, n_neigh, n_neigh0, ef_build, ef_search, has_deletion)
        self._args = {
            "dimension": dimension,
            "dtype": dtype,
            "max_item": max_item,
            "n_neigh": n_neigh,
            "n_neigh0": n_neigh0,
            "ef_build": ef_build,
            "ef_search": ef_search,
            "has_deletion": has_deletion
        }


class PQIndex(HoraANNIndex):
    """PQIndex

    the implementation of algorithm https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf
    """

    def __init__(self, dimension, dtype, n_sub=4, sub_bits=4, train_epoch=100):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraPQIndexUsize
        elif dtype == "str":
            self.ann_type = HoraPQIndexStr
        else:
            raise TypeError("invalid dtype {}, only accept {}".format(
                dtype, self._accept_dtype))
        self.ann_idx = self.ann_type(
            dimension, n_sub, sub_bits, train_epoch)
        self._args = {
            "dimension": dimension,
            "dtype": dtype,
            "n_sub": n_sub,
            "sub_bits": sub_bits,
            "train_epoch": train_epoch
        }


class SSGIndex(HoraANNIndex):
    """SSGIndex

    the implementation of algorithm https://arxiv.org/abs/1907.06146
    """

    def __init__(self, dimension, dtype, neighbor_neighbor_size=100, init_k=100, index_size=100, angle=60.0, root_size=100):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraSSGIndexUsize
        elif dtype == "str":
            self.ann_type = HoraSSGIndexStr
        else:
            raise TypeError("invalid dtype {}, only accept {}".format(
                dtype, self._accept_dtype))
        self.ann_idx = self.ann_type(
            dimension, neighbor_neighbor_size, init_k, index_size, angle, root_size)

        self._args = {
            "dimension": dimension,
            "dtype": dtype,
            "neighbor_neighbor_size": neighbor_neighbor_size,
            "init_k": init_k,
            "index_size": index_size,
            "angle": angle,
            "root_size": root_size,
        }


class IVFPQIndex(HoraANNIndex):
    """IVFPQIndex

    the implementation of algorithm https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf
    """

    def __init__(self, dimension, dtype, n_sub=25, sub_bits=4, n_kmeans_center=256, search_n_center=8, train_epoch=100):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraIVFPQIndexUsize
        elif dtype == "str":
            self.ann_type = HoraIVFPQIndexStr
        else:
            raise TypeError("invalid dtype {}, only accept {}".format(
                dtype, self._accept_dtype))
        self.ann_idx = self.ann_type(
            dimension, n_sub, sub_bits, n_kmeans_center, search_n_center, train_epoch)

        self._args = {
            "dimension": dimension,
            "dtype": dtype,
            "n_sub": n_sub,
            "sub_bits": sub_bits,
            "n_kmeans_center": n_kmeans_center,
            "search_n_center": search_n_center,
            "train_epoch": train_epoch,
        }
