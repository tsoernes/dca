from pymongo import MongoClient
from bson import SON
client = MongoClient('localhost', 1234, document_class=SON, w=1, j=True)
db = client['qlearnnet-net_lr-net_lr_decay2']
