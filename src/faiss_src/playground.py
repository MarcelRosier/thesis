import numpy as np
from numpy.core.numeric import count_nonzero
from numpy.lib.shape_base import tile
from numpy.testing._private.utils import print_assert_equal
import faiss
import os
from utils import load_real_tumor

dimension = 128 * 128 * 128  # dimensions of each vector
# number of vectors #! dont use ~ 1000, CRASHES YOUR PC (for 16GB RAM)
# n = 50000  # ! danger zone: 100000
n = 200

SYN_TUMOR_BASE_PATH = '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset'
SYN_TUMOR_PATH_TEMPLATE = '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset/{id}/Data_0001.npz'


def create_and_store_index(path, is_test=True):
    # np.random.seed(1)
    # db_vectors = np.random.random((n, dimension)).astype('float32')
    # # map to 0/ 1
    # db_vectors[db_vectors < 0.5] = 0
    # db_vectors[db_vectors >= 0.5] = 1
    # build vector db

    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders.sort(key=lambda f: int(f))

    # only get a subset of the data if its a test
    if is_test:
        folders = folders[:n]
    print("creating vector db...")
    vectors_list = [get_vector_from_tumor_data(
        tumor_id=f) for f in folders]
    db_vectors = np.asarray(vectors_list, dtype=np.float32)
    print("finished creating vector db...")

    nlist = 5  # number of clusters
    quantiser = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)

    print(index.is_trained)   # False
    index.train(db_vectors)  # train on the database vectors
    print(index.ntotal)   # 0
    index.add(db_vectors)   # add the vectors and update the index
    print(index.ntotal)   # 200

    # save the index to disk
    faiss.write_index(
        index, path)


def load_index(path):
    return faiss.read_index(path)


def execute_query(index, query=None):
    n_query = 1
    k = 10  # return 3 nearest neighbours
    np.random.seed(0)
    query_vectors = query.astype('float32')
    if query is None:
        query_vectors = np.random.random(
            (n_query, dimension)).astype('float32')
        query_vectors[query_vectors < 0.5] = 0
        query_vectors[query_vectors >= 0.5] = 1
    else:
        pass
    print(query_vectors.shape)
    distances, indices = index.search(query_vectors, k)
    print(distances)
    print(indices)


def get_vector_from_tumor_data(tumor_id):
    tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(id=tumor_id))['data']
    if tumor.shape != (128, 128, 128):
        tumor = np.delete(np.delete(
            np.delete(tumor, 128, 0), 128, 1), 128, 2)
    # normalize
    max_val = tumor.max()
    if max_val != 0:
        tumor *= 1.0/max_val
    # threshold
    tumor_02 = np.copy(tumor)
    tumor_02[tumor_02 < 0.2] = 0
    tumor_02[tumor_02 >= 0.2] = 1

    # transform to vector
    tumor_02_vector = tumor_02.flatten()
    return tumor_02_vector


def load_query(path):
    (t1c, flair) = load_real_tumor(path)
    t1c_vector = t1c.flatten()
    t1c_vector = t1c_vector.reshape(1, len(t1c_vector))
    # only consider t1c for now
    return t1c_vector


def run():
    # get_vector_from_tumor_data(
    #     path='/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset/0/Data_0001.npz')
    # create_and_store_index(
    #     path="/home/marcel/Projects/uni/thesis/src/faiss_src/indices/200_02_vector.index", is_test=True)
    index = load_index(
        path="/home/marcel/Projects/uni/thesis/src/faiss_src/indices/200_02_vector.index")
    query = load_query(
        path='/home/marcel/Projects/uni/thesis/real_tumors/tgm001_preop')
    execute_query(index=index, query=query)
