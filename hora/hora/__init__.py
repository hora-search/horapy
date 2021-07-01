import numpy
from .hora import BruteForceIndex as HoraBruteForceIndex
from .hora import BPTIndex as HoraBPTIndex
from .hora import HNSWIndex as HoraHNSWIndex
from .hora import PQIndex as HoraPQIndex
from .hora import SSGIndex as HoraSSGIndex
from .hora import IVFPQIndex as HoraIVFPQIndex


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
        self.ann_type = HoraBruteForceIndex
        self.ann_idx = HoraBruteForceIndex(dimension)


# class BPTIndex(ANNIndex):
#     def __init__(self, dimension, tree_num, candidate_size, dtype):
#         super().__init__(dimension, dtype)
#         self.ann_type = HoraBPTIndex
#         self.ann_idx = HoraBPTIndex(dimension, tree_num, candidate_size)


class HNSWIndex(HoraANNIndex):
    def __init__(self, dimension, max_item, n_neigh, n_neigh0, ef_build, ef_search, has_deletion, dtype):
        super().__init__(dimension, dtype)
        self.ann_type = HoraHNSWIndex
        self.ann_idx = HoraHNSWIndex(
            dimension, max_item, n_neigh, n_neigh0, ef_build, ef_search, has_deletion)


class PQIndex(HoraANNIndex):
    def __init__(self, dimension, n_sub, sub_bits, train_epoch, dtype):
        super().__init__(dimension, dtype)
        self.ann_type = HoraPQIndex
        self.ann_idx = HoraPQIndex(dimension, n_sub, sub_bits, train_epoch)


class SSGIndex(HoraANNIndex):
    def __init__(self, dimension, neighbor_neighbor_size, init_k, index_size, angle, root_size, dtype):
        super().__init__(dimension, dtype)
        self.ann_type = HoraSSGIndex
        self.ann_idx = HoraSSGIndex(
            dimension, neighbor_neighbor_size, init_k, index_size, angle, root_size)


class IVFPQIndex(HoraANNIndex):
    def __init__(self, dimension, n_sub, sub_bits, n_kmeans_center, search_n_center, train_epoch, dtype):
        super().__init__(dimension, dtype)
        self.ann_type = HoraIVFPQIndex
        self.ann_idx = HoraIVFPQIndex(
            dimension, n_sub, sub_bits, n_kmeans_center, search_n_center, train_epoch)
