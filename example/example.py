import horapy
import random
from annoy import AnnoyIndex
import random
from tqdm import tqdm


def make_indices(dimensions):
    indices = []
    indices.append(horapy.BruteForceIndex(dimensions, "usize"))
    indices.append(horapy.HNSWIndex(dimensions,"usize"))
    # indices.append(horapy.PQIndex(dimensions,"usize"))
    indices.append(horapy.SSGIndex(
        dimensions, "usize"))
    # indices.append(horapy.IVFPQIndex(dimensions,"usize"))

    return indices


def make_test_data(dimensions, indices):
    round = 1000
    float_list = []
    for i in range(round):
        base = random.randrange(1000)
        v = [random.gauss(0, 10) + base for z in range(dimensions)]
        float_list.append(v)
    features = []
    for i in range(round):
        f = []
        for j in range(dimensions):
            f.append(float_list[i][j])
        features.append((f, i))
    for idx in indices:
        for i in features:
            idx.add(i[0], i[1])
    return features


def make_annoy_idx(dimensions, features):
    t = AnnoyIndex(dimensions, 'euclidean')
    for f in features:
        t.add_item(int(f[1]), f[0])
    t.build(10)
    return t


def main():
    dimensions = 10
    indices = make_indices(dimensions)
    features = make_test_data(dimensions, indices)

    for idx in indices:
        idx.build("euclidean")

    annoy_idx = make_annoy_idx(dimensions, features)

    bpforest = 0
    pq = 0
    hnsw = 0
    ssg = 0
    annoy = 0

    round = 1000
    k = 100
    for i in tqdm(range(round)):
        idx = features[random.randrange(0, len(features))]
        print(indices[0].search(idx[0], k))
        base_result = [item[0]
                       for item in indices[0].search(idx[0], k)]

        result = [item[0] for item in indices[1].search(idx[0], k)]
        bpforest += sum([item in base_result for item in result])

        result = [item[0] for item in indices[2].search(idx[0], k)]
        hnsw += sum([item in base_result for item in result])

        # result = [item[0].idx for item in indices[3].search(idx[0], k)]
        # pq += sum([item in base_result for item in result])

        result = [item[0].idx for item in indices[4].search(idx[0], k)]
        ssg += sum([item in base_result for item in result])

        result = [item
                  for item in annoy_idx.get_nns_by_item(int(idx[1]), k)]
        annoy += sum([str(item) in base_result for item in result])

    # print("bp {} \n pq {} \n hnsw {} \n ssg {} \n annoy {}".format(
    #     bpforest/float(round), pq/float(round), hnsw/float(round), ssg/float(round), annoy/float(round)))


if __name__ == "__main__":
    main()
