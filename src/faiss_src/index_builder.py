import os

import faiss
import numpy as np
from faiss.contrib.ondisk import merge_ondisk

dimension = 128 * 128 * 128  # dimensions of each vector
# number of vectors #! dont use ~ 1000, CRASHES YOUR PC (for 16GB RAM)
# n = 50000  # ! danger zone: 100000
n = 200

SYN_TUMOR_BASE_PATH = '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset'
SYN_TUMOR_PATH_TEMPLATE = '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset/{id}/Data_0001.npz'
TUMOR_SUBSET_TESTING_SIZE = 512
TUMOR_SUBSET_BENCHMARK_SIZE = 50000
REAL_TUMOR_BASE_PATH_TEMPLATE = '/home/marcel/Projects/uni/thesis/real_tumors/{id}'

INDEX_BASE_PATH = '/home/marcel/Projects/uni/thesis/src/faiss_src/indices/'


def get_vector_from_tumor_data(tumor_id):
    tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(id=tumor_id))['data']

    # crop to 128^3
    if tumor.shape != (128, 128, 128):
        tumor = np.delete(np.delete(
            np.delete(tumor, 128, 0), 128, 1), 128, 2)

    # normalize
    max_val = tumor.max()
    if max_val != 0:
        tumor *= 1.0/max_val

    # threshold for t1c
    tumor_02 = np.copy(tumor)
    tumor_02[tumor_02 < 0.2] = 0
    tumor_02[tumor_02 >= 0.2] = 1

    # transform to vector
    return tumor_02.flatten()


def create_index_base(base_vectors):
    nlist = 4  # number of clusters
    quantiser = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)
    index.train(base_vectors)
    faiss.write_index(index, INDEX_BASE_PATH + "trained.index")


def add_vectors_to_index(index_id, vectors):

    # load trained index
    index = faiss.read_index()

def get_tumors_as_vector_list(range_start=0, range_end=250, is_test=False):
    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders.sort(key=lambda f: int(f))

    # only get a subset of the data if its a test
    if is_test:
        folders = folders[:TUMOR_SUBSET_TESTING_SIZE]
    else:
        folders = folders[:TUMOR_SUBSET_BENCHMARK_SIZE]

    folders = folders[range_start:range_end]

    vectors_list = []
    id_index_map = {}
    for i, f in enumerate(folders):
        vectors_list.append(get_vector_from_tumor_data(tumor_id=f))
        id_index_map[i] = f

    # db_vectors = np.asarray(vectors_list, dtype=np.float32)
    # TODO manage index map
    return vectors_list


def build_index():
    print("load vectors")
    vectors_list = get_tumors_as_vector_list(range_start=0, range_end=250)
    print("transform to array")
    db_vectors_1 = np.asarray(vectors_list[:256], dtype=np.bool)
    del vectors_list
    print("create index")
    create_index_base(db_vectors_1)
    del db_vectors_1

    # db_vectors_2 = np.asarray(vectors_list[256:], dtype=np.float32)
