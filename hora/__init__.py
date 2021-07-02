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
    def __init__(self, dimension, dtype):
        self.ann_idx = None
        self.dtype = dtype
        self.ann_type = None
        self.dimension = dimension

    def __str__(self):
        return "Hora ANNIndex <{}>".format(self.name())

    def build(self, metrics=""):
        self.ann_idx.build(metrics)

    def add(self, vs, idx=None):
        if isinstance(vs, list):
            self.ann_idx.add(vs, idx)

    def search(self, vs, k):
        if isinstance(vs, numpy.ndarray):
            return self.ann_idx.search_np(vs, k)
        elif isinstance(vs, list):
            return self.ann_idx.search_k(vs, k)

    def name(self):
        return self.ann_idx.name()

    def load(self, path):
        self.ann_idx = self.ann_type.load(path)

    def dump(self, path):
        self.ann_idx.dump(path)


class BruteForceIndex(HoraANNIndex):
    def __init__(self, dimension, dtype):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraBruteForceIndexUsize
            self.ann_idx = HoraBruteForceIndexUsize(dimension)
        elif dtype == "str":
            self.ann_type = HoraBruteForceIndexStr
            self.ann_idx = HoraBruteForceIndexStr(dimension)
        else:
            raise ValueError("invalid dtype {}".format(dtype))


# class BPTIndex(ANNIndex):
#     def __init__(self, dimension, tree_num, candidate_size, dtype):
#         super().__init__(dimension, dtype)
#         self.ann_type = HoraBPTIndex
#         self.ann_idx = HoraBPTIndex(dimension, tree_num, candidate_size)


class HNSWIndex(HoraANNIndex):
    def __init__(self, dimension, max_item, n_neigh, n_neigh0, ef_build, ef_search, has_deletion, dtype):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraHNSWIndexUsize
            self.ann_idx = HoraHNSWIndexUsize(
                dimension, max_item, n_neigh, n_neigh0, ef_build, ef_search, has_deletion)
        elif dtype == "str":
            self.ann_type = HoraHNSWIndexStr
            self.ann_idx = HoraHNSWIndexUsize(
                dimension, max_item, n_neigh, n_neigh0, ef_build, ef_search, has_deletion)
        else:
            raise ValueError("invalid dtype {}".format(dtype))


class PQIndex(HoraANNIndex):
    def __init__(self, dimension, n_sub, sub_bits, train_epoch, dtype):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraPQIndexUsize
            self.ann_idx = HoraPQIndexUsize(
                dimension, n_sub, sub_bits, train_epoch)
        elif dtype == "str":
            self.ann_type = HoraPQIndexStr
            self.ann_idx = HoraPQIndexStr(
                dimension, n_sub, sub_bits, train_epoch)
        else:
            raise ValueError("invalid dtype {}".format(dtype))


class SSGIndex(HoraANNIndex):
    def __init__(self, dimension, neighbor_neighbor_size, init_k, index_size, angle, root_size, dtype):
        super().__init__(dimension, dtype)
        if dtype == "usize":
            self.ann_type = HoraSSGIndexUsize
            self.ann_idx = HoraSSGIndexUsize(
                dimension, neighbor_neighbor_size, init_k, index_size, angle, root_size)
        elif dtype == "str":
            self.ann_type = HoraSSGIndexStr
            self.ann_idx = HoraSSGIndexStr(
                dimension, neighbor_neighbor_size, init_k, index_size, angle, root_size)
        else:
            raise ValueError("invalid dtype {}".format(dtype))


class IVFPQIndex(HoraANNIndex):
    def __init__(self, dimension, n_sub, sub_bits, n_kmeans_center, search_n_center, train_epoch, dtype):
        super().__init__(dimension, dtype)
        self.ann_type = HoraIVFPQIndex
        self.ann_idx = HoraIVFPQIndex(
            dimension, n_sub, sub_bits, n_kmeans_center, search_n_center, train_epoch)
        if dtype == "usize":
            self.ann_type = HoraIVFPQIndexUsize
            self.ann_idx = HoraIVFPQIndexUsize(
                dimension, n_sub, sub_bits, n_kmeans_center, search_n_center, train_epoch)
        elif dtype == "str":
            self.ann_type = HoraIVFPQIndexStr
            self.ann_idx = HoraIVFPQIndexStr(
                dimension, n_sub, sub_bits, n_kmeans_center, search_n_center, train_epoch)
        else:
            raise ValueError("invalid dtype {}".format(dtype))
