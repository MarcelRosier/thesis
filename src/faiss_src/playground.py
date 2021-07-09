import json
import logging
import os
from datetime import datetime

from numpy.lib import utils

import faiss
import numpy as np
from utils import load_real_tumor
import utils

dimension = 128 * 128 * 128  # dimensions of each vector
# number of vectors #! dont use ~ 1000, CRASHES YOUR PC (for 16GB RAM)
# n = 50000  # ! danger zone: 100000
n = 200

SYN_TUMOR_BASE_PATH = '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset'
SYN_TUMOR_PATH_TEMPLATE = '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset/{id}/Data_0001.npz'
TUMOR_SUBSET_TESTING_SIZE = 200
REAL_TUMOR_BASE_PATH_TEMPLATE = '/home/marcel/Projects/uni/thesis/real_tumors/{id}'


def create_and_store_index(index_path, map_path=None, is_test=True):
    """
    creates the faiss index and stores it to @path
    @test specifies if the entres tumor data or only a subste should be used
    returns a dict that maps index id to tumor folder ids
    """
    # build vector db

    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders.sort(key=lambda f: int(f))

    # only get a subset of the data if its a test
    if is_test:
        folders = folders[:TUMOR_SUBSET_TESTING_SIZE]

    logging.info("creating vector db...")
    vectors_list = []
    id_index_map = {}
    for i, f in enumerate(folders):
        vectors_list.append(get_vector_from_tumor_data(tumor_id=f))
        id_index_map[i] = f

    db_vectors = np.asarray(vectors_list, dtype=np.float32)

    logging.info("finished creating vector db...")

    nlist = 5  # number of clusters
    quantiser = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)

    index.train(db_vectors)  # train on the database vectors
    index.add(db_vectors)   # add the vectors and update the index
    logging.info("total vectors: {}".format(index.ntotal))   # 200

    # save the index to disk
    faiss.write_index(
        index, index_path)
    logging.info("Saved index to: {}".format(index_path))

    if map_path is not None:
        with open(map_path.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "w") as file:
            json.dump(id_index_map, file)

    return id_index_map


def load_index(path):
    return faiss.read_index(path)


def execute_query(index, query=None):
    k = 10  # return 10 nearest neighbours
    query_vectors = query.astype('float32')
    distances, indices = index.search(query_vectors, k)
    logging.info("distances: {}".format(distances))
    logging.info("indices: {}".format(indices))
    return distances, indices


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


def generate_query(path):
    (t1c, flair) = load_real_tumor(path)
    t1c_vector = t1c.flatten()
    t1c_vector = t1c_vector.reshape(1, len(t1c_vector))
    # only consider t1c for now
    return t1c_vector


def run(real_tumor):
    # create_and_store_index(
    #     index_path="/home/marcel/Projects/uni/thesis/src/faiss_src/indices/200_02_vector.index",
    #     # map_path='/home/marcel/Projects/uni/thesis/src/faiss_src/indices/{}_id_index_map.json',
    #     is_test=True)
    index = load_index(
        path="/home/marcel/Projects/uni/thesis/src/faiss_src/indices/200_02_vector.index")
    query = generate_query(
        path=REAL_TUMOR_BASE_PATH_TEMPLATE.format(id=real_tumor))
    D, I = execute_query(index=index, query=query)
    return D, I
    # [for i in I]
