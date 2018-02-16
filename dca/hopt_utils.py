import pickle
from operator import itemgetter

from bson import SON
from hyperopt.mongoexp import MongoTrials
from pymongo import MongoClient

port = 1234
"""
NOTE For some reason, 'db.col.someFunc' is not the same as
'col = db['colname']; col.someFunc'. The latter seem to work, the former does not.
"""


def hopt_results(pp, trials):
    """Gather losses and corresponding params for valid results"""
    if type(trials) is MongoTrials:
        uri = pp['hopt_fname'].replace('mongo:', '')
        attachments = get_pps_mongo(uri)
        valid = list(filter(lambda x: x['result']['status'] == 'ok', trials.trials))
        valid_results = []
        pkeys = valid[0]['misc']['vals'].keys()
        params = {key: [] for key in pkeys}
        for i, res in enumerate(valid):
            valid_results.append((res['result']['loss'], i))
            for pkey in pkeys:
                params[pkey].append(res['misc']['vals'][pkey][0])
    else:
        attachments = trials.attachments
        results = [(e['loss'], i, e['status']) for i, e in enumerate(trials.results)]
        valid_results = filter(lambda x: x[2] == 'ok', results)
        params = trials.vals
    return valid_results, params, attachments


def hopt_trials(pp):
    """Load trials from MongoDB or Pickle file, depending on 'hopt_fname'"""
    f_name = pp['hopt_fname']
    try:
        if f_name.startswith("mongo"):
            # e.g. 'mongo://localhost:1234/results-singhnet-net_lr-beta/jobs'
            f_name = f"mongo://localhost:1234/{f_name.replace('mongo:', '')}/jobs"
            print(f"Attempting to connect to mongodb with url {f_name}")
            return MongoTrials(f_name)
        else:
            f_name = pp['hopt_fname'].replace('.pkl', '') + '.pkl'
            with open(f_name, "rb") as f:
                return pickle.load(f)
    except FileNotFoundError:
        print(f"Could not find {f_name}.")
        raise
    except:
        print("Have you started mongod server in 'db' dir? \n"
              "mongod --dbpath . --directoryperdb"
              " --journal --nohttpinterface --port 1234")
        raise


def mongo_decide_gpu_usage(mongo_uri, max_gpu_procs):
    """Decide whether or not to use GPU based on how many proceses currently use it"""
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


def add_pp_pickle(trials, pp):
    """Add problem params to Trials object attachments, if it differs from
    the last one added"""
    att = trials.attachments
    if "pp0" not in att:
        att['pp0'] = pp
        return
    n = 0
    while f"pp{n}" in att:
        n += 1
    n -= 1
    if att[f'pp{n}'] != pp:
        att[f'pp{n+1}'] = pp


def add_pp_mongo(mongo_uri, pp):
    """Add problem params to mongo db, if it differs from
    one already added"""
    # TODO should add if it differs from the last one added
    client = MongoClient('localhost', port, document_class=SON, w=1, j=True)
    db = client[mongo_uri]
    col = db['pp']
    doc = col.find_one(pp)
    if doc is None:
        col.insert_one(pp)
    client.close()


def get_pps_mongo(mongo_uri):
    """Get problems params stored in mongo db, sorted by the time they were added"""
    # Does not use attachmens object, will store different loc
    client = MongoClient('localhost', port, document_class=SON, w=1, j=True)
    db = client[mongo_uri]
    col = db['pp']
    pps = []
    for ppson in col.find():
        pp = ppson.to_dict()
        pp['dt'] = pp['_id'].generation_time
        pps.append(pp)
    pps.sort(key=itemgetter('dt'))
    client.close()
    return pps


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
        if dbname in ['admin', 'local']:
            continue
        db = client[dbname]
        gpup = db['gpu_procs'].find_one()
        if gpup is not None:
            gpup = gpup['gpu_procs']
        print(f"DB: {dbname}, GPU proc count: {gpup}")
        for colname in db.list_collection_names():
            col = db[colname]
            print(f"  Col: {colname}, count: {col.count()}")
    client.close()
