from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES
from pprint import pprint

es = Elasticsearch()
ses = SignatureES(es)

path_1 = '/home/marcel/Projects/uni/thesis/src/image_match/example_images/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg'
path_2 = '/home/marcel/Projects/uni/thesis/src/image_match/example_images/ml_2.jpeg'
path_3 = '/home/marcel/Projects/uni/thesis/src/image_match/example_images/ml_cat.jpg'
path_4 = '/home/marcel/Projects/uni/thesis/src/image_match/example_images/ml_duck.jpg'
# ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
# ses.add_image(path_2)
# ses.delete_duplicates(path)

# ses.add_image(
#     'https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Example.svg/600px-Example.svg.png')
# ses.add_image(path_3)
# ses.add_image(
#     'https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
# ses.add_image('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')


def add_images():
    ses.add_image(path_1)
    ses.add_image(path_2)
    ses.add_image(path_3)
    ses.add_image(path_4)


def delete_duplicates():
    ses.delete_duplicates(path_1)
    ses.delete_duplicates(path_2)
    ses.delete_duplicates(path_3)
    ses.delete_duplicates(path_4)


def query():
    res = ses.search_image(path_4)
    print("#results: ", len(res))

    res = [(e['dist'], e['path']) for e in res]
    pprint(res)


# add_images()
delete_duplicates()
query()
