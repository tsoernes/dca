import imp

from bson import SON
from pymongo import MongoClient

client = MongoClient('localhost', 1234, document_class=SON, w=1, j=True)
db = client['qlearnnet-net_lr-net_lr_decay2']
rel = imp.reload
