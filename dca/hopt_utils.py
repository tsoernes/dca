from bson import SON
from pymongo import MongoClient

port = 1234

"""
NOTE For some reason, 'db.col.someFunc' is not the same as
'col = db['colname']; col.someFunc'. The latter seem to work, the former does not.
"""


def mongo_decide_gpu_usage(mongo_uri, max_gpu_procs):
    # The DB should contain a collection 'gpu_procs' with one document,
    # {'gpu_procs': N}, where N is the current number of procs that utilize the GPU.
    client = MongoClient('localhost', port, document_class=SON, w=1, j=True)
    db = client[mongo_uri]
    col = db['gpu_procs']
    doc = col.find_one()
    if doc is None:
        col.insert_one({'gpu_procs': 0})
        doc = col.find_one()
    if doc['gpu_procs'] >= max_gpu_procs:
        print("MONGO decided not to use GPU")
        using_gpu = False
    else:
        print("MONGO increasing GPU proc count")
        col.find_one_and_update(doc, {'$inc': {'gpu_procs': 1}})
        using_gpu = True
    client.close()
    return using_gpu


def mongo_decrease_gpu_procs(mongo_uri):
    """Decrease GPU process count in Mongo DB. Fails if none are in use."""
    client = MongoClient('localhost', port, document_class=SON, w=1, j=True)
    db = client[mongo_uri]
    col = db['gpu_procs']
    doc = col.find_one()
    assert doc is not None
    assert doc['gpu_procs'] > 0
    print("MONGO decreaseing GPU proc count")
    col.find_one_and_update(doc, {'$inc': {'gpu_procs': -1}})
    client.close()


def mongo_get_pp(mongo_uri):
    raise NotImplementedError
    # Does not use attachmens object, will store different loc
    client = MongoClient('localhost', port, document_class=SON, w=1, j=True)
    db = client[mongo_uri]
    col = db['pp']
    doc = col.find_one()
    client.close()
    return doc


def mongo_prune_suspended(mongo_uri):
    """Remove jobs with status 'suspended'."""
    client = MongoClient('localhost', port, document_class=SON, w=1, j=True)
    db = client[mongo_uri]
    col = db['jobs']
    pre_count = col.count()
    col.delete_many({'results': {'status': 'suspended'}})
    count = col.count()
    print(f"Deleted {pre_count-count}, suspended jobs, current count {count}")
    client.close()


def mongo_list_dbs():
    """List all databases, their GPU process count and collections."""
    client = MongoClient('localhost', port)
    for dbname in client.list_database_names():
        db = client[dbname]
        print(f"DB: {db}, GPU proc count {db['gpu_procs']['gpu_procs']}")
        for colname in db.list_collection_names():
            col = db[colname]
            print(f"  Col: {colname}, count: {col.count()}")
    client.close()
