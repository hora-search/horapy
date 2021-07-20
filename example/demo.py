import numpy as np
from hora import HNSWIndex

dimension = 50
n = 1000
index = HNSWIndex(dimension, "usize")
samples = np.float32(np.random.rand(n, dimension))
for i in range(0, len(samples)):
    index.add(np.float32(samples[i]), i)
index.build("euclidean")
target = np.random.randint(0, n)
print("{} has neighbors: {}".format(target, index.search(samples[target], 10))) # 631 has neighbors: [631, 656, 113, 457, 584, 179, 586, 979, 619, 976]
