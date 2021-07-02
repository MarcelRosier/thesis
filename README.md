# thesis

## image-match
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
