# thesis

## FAISS

### Setup
- Create a new conda env   
`conda create --name myenv`
- Install faiss-cpu (or gpu if applicable)
    ```shell
    # CPU-only version
    conda install -c pytorch faiss-cpu

    # GPU(+CPU) version
    conda install -c pytorch faiss-gpu

    # or for a specific CUDA version
    conda install -c pytorch faiss-gpu cudatoolkit=10.2 # for CUDA 10.2
    ```
- Troubleshooting:
    - MKL Errors:  
    A few to try when running into *Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.* or similar:  
    ```shell
    # Add nomkl
    conda install nomkl numpy scipy scikit-learn numexpr
    conda remove mkl mkl-service

    # Remove it adding didnt help
    conda remove nomkl

    # Not sure why, but helps sometimes
    conda install  -f  numpy
    ```

### Usage
- Create Index
    ```python
    dimension = 128 * 128  # dimensions of each vector
    n = 50000   # number of vectors
    np.random.seed(1)
    db_vectors = np.random.random((n, dimension)).astype('float32')

    nlist = 5  # number of clusters
    quantiser = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)

    index.train(db_vectors)  # train on the database vectors
    index.add(db_vectors)   # add the vectors and update the index
    ```
- Save/ Load the index
    ```python
    faiss.write_index(index, "vector.index")
    index = faiss.read_index("vector.index")
    ```   
- Query
    ```python
    n_query = 10
    k = 3  # return 3 nearest neighbours
    np.random.seed(0)
    query_vectors = np.random.random((n_query, dimension)).astype('float32')
    distances, indices = index.search(query_vectors, k)
    ```   

---  

## IMAGE-MATCH
![](https://cloud.githubusercontent.com/assets/6517700/17741093/41040a64-649b-11e6-8499-48b78ddca56b.png
)
### Setup
- Make sure Elasticsearch is installed and running  
`sudo service elasticsearch status`  
if not running start it  
`sudo service elasticsearch start`  
if not installed refer to [this install guide](https://www.elastic.co/de/downloads/elasticsearch)

- Setup a virtual ennvironment for python  
 `virtualenv -p python3.5 image-match-env/`  
 python 3.5.10 seems to work, higher version are apparently not compatible

- Activate the venv and install all relevant libraries via pip `pip install -r requirements.txt`
    ```
    numpy
    scikit-image>=0.14
    elasticsearch<6.0.0,>=5.0.0
    image_match
    ```
- Now you should be able to use image-match

### Usage
- Init
    ```python
    from elasticsearch import Elasticsearch
    from image_match.elasticsearch_driver import SignatureES

    es = Elasticsearch()
    ses = SignatureES(es)
    ```
- Common methods  

    ```python
    ses.add_image(path) # adds iamge to the database
    ses.delete_duplicates(path) # deletes all except one entry with the same !path!
    ses.search_image(path,[all_orientations=True]) 
    # returns list of matching images 
    # [{'dist': 0.0, 'metadata': None, 'path': 'path', 'id': 'id', 'score': 63.0}]
    ```
