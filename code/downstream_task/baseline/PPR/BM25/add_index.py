from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm


data_dir = "../../../../../datasets/"
train = json.load(open(os.path.join(data_dir, "PMC-Patients_train.json"), "r"))
dev = json.load(open(os.path.join(data_dir, "PMC-Patients_dev.json"), "r"))
test = json.load(open(os.path.join(data_dir, "PMC-Patients_test.json"), "r"))

es = ES()
train_body = []
dev_body = []
test_body = []
for case in tqdm(train):
    train_body.append({"index": {"_index": "patient_train", "_id": case['patient_uid']}})
    train_body.append({"patient": case['patient']})

    if len(train_body) >= 10000:
        es.bulk(train_body)
        train_body = []

es.bulk(train_body)

for case in tqdm(dev):
    dev_body.append({"index": {"_index": "patient_dev", "_id": case['patient_uid']}})
    dev_body.append({"patient": case['patient']})

for case in tqdm(test):
    test_body.append({"index": {"_index": "patient_test", "_id": case['patient_uid']}})
    test_body.append({"patient": case['patient']})

es.bulk(dev_body)
es.bulk(test_body)
print(es.count(index = 'patient_train'))
print(es.count(index = 'patient_dev'))
print(es.count(index = 'patient_test'))

#import ipdb; ipdb.set_trace()