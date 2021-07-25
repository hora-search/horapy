<div align="center">
  <img src="asset/logo.svg" width="70%"/>
</div>

# horapy

**[[Homepage](http://horasearch.com/)]** **[[Document](https://horasearch.com/doc)]** **[[Examples](https://horasearch.com/doc/example.html)]** **[[Hora](https://github.com/hora-search/hora)]** 

Python bidding for the **`Hora Approximate Nearest Neighbor Search`**

# Key Features

* **Performant** ⚡️
  * **SIMD-Accelerated ([packed_simd](https://github.com/rust-lang/packed_simd))**
  * **Stable algorithm implementation**
  * **Multiple threads design**

* **Multiple Indexes Support** 🚀
  * `Hierarchical Navigable Small World Graph Index(HNSWIndex)` ([detail](https://arxiv.org/abs/1603.09320))
  * `Satellite System Graph (SSGIndex)` ([detail](https://arxiv.org/abs/1907.06146))
  * `Product Quantization Inverted File(PQIVFIndex)` ([detail](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf))
  * `Random Projection Tree(RPTIndex)` (LSH, WIP)
  * `BruteForce (BruteForceIndex)` (naive implementation with SIMD)

* **Portable** 💼
  * Support `no_std` (WIP, partial)
  * Support `Windows`, `Linux` and `OS X`
  * Support `IOS` and `Android` (WIP)
  * **No** heavy dependency, such as `BLAS`

* **Reliability** 🔒
  * `Rust` compiler secure all code
  * Memory managed by `Rust`
  * Broad testing coverage

* **Multiple Distances Support** 🧮
  * `Dot Product Distance`
    * ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Csum%7B%28x*y%29%7D)
  * `Euclidean Distance`
    * ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Csqrt%7B%5Csum%7B%28x-y%29%5E2%7D%7D)
  * `Manhattan Distance`
    * ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Csum%7B%7C%28x-y%29%7C%7D)
  * `Cosine Similarity`
    * ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Cfrac%7Bx%20*y%7D%7B%7C%7Cx%7C%7C*%7C%7Cy%7C%7C%7D)

* **Productive** ⭐
  * Well documented
  * Elegant and simple API, easy to learn

# Benchmark
<img src="asset/fashion-mnist-784-euclidean_10_euclidean.png"/>

by `aws t2.medium (CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz)` [more information](https://github.com/hora-search/ann-benchmarks)

## Installation

```bash
pip install horapy
```

## Example

```Python
import numpy as np
from horapy import HNSWIndex

dimension = 50
n = 1000

# init index instance
index = HNSWIndex(dimension, "usize")

samples = np.float32(np.random.rand(n, dimension))
for i in range(0, len(samples)):
    # add node
    index.add(np.float32(samples[i]), i)

index.build("euclidean")  # build index

target = np.random.randint(0, n)
# 410 in Hora ANNIndex <HNSWIndexUsize> (dimension: 50, dtype: usize, max_item: 1000000, n_neigh: 32, n_neigh0: 64, ef_build: 20, ef_search: 500, has_deletion: False)
# has neighbors: [410, 736, 65, 36, 631, 83, 111, 254, 990, 161]
print("{} in {} \nhas neighbors: {}".format(
    target, index, index.search(samples[target], 10)))  # search
```

# License

The entire repo is under [Apache License](https://github.com/hora-search/hora/blob/main/LICENSE).