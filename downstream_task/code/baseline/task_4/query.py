from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm
import numpy as np
from utils import getNDCG, getPrecision, getRR, getRecall


data_dir = "../../../datasets/task_3_patient2patient_retrieval"
train_data = json.load(open(os.path.join(data_dir, "patient2patient_train.json"), "r"))
dev_data = json.load(open(os.path.join(data_dir, "patient2patient_dev.json"), "r"))
test_data = json.load(open(os.path.join(data_dir, "patient2patient_test.json"), "r"))
print("Load Done")
es = ES(http_auth = "elastic:UpMvQdoGeuoHDfrv00qF", maxsize = 40)
'''
precisions = []
NDCGs = []
RRs = []
recalls = []
for patient_id in train_data:
    query_text = es.get(index = "patient_train", id = patient_id)['_source']['patient']
    if len(query_text) > 1024:
        query_text = query_text[:1024]
    query = {"query": {"match": {"patient": query_text}}}
    results = es.search(body = query, index = "patient_train", size = 11, _source = False)
    result_ids = [x['_id'] for x in results['hits']['hits'][1:]]
    golden_list = train_data[patient_id][0] + train_data[patient_id][1]
    precision = getPrecision(golden_list, result_ids)
    recall = getRecall(golden_list, result_ids)
    NDCG = getNDCG(train_data[patient_id][0], train_data[patient_id][1], result_ids)
    precisions.append(precision)
    NDCGs.append(NDCG)
    recalls.append(recall)
    
    if precision == 0:
        results = es.search(body = query, index = "patient_train", size = 10000, _source = False)
        result_ids = [x['_id'] for x in results['hits']['hits']]
        RR = getRR(golden_list, result_ids)
    else:
        RR = getRR(golden_list, result_ids)
    RRs.append(RR)

print("=========Train=========")
print(np.mean(RRs), np.mean(precisions), np.mean(recalls), np.mean(NDCGs))
'''

precisions = []
NDCGs = []
RRs = []
recalls = []
for patient_id in tqdm(dev_data):
    query_text = es.get(index = "patient_dev", id = patient_id)['_source']['patient']
    if len(query_text) > 1024:
        query_text = query_text[:1024]
    query = {"query": {"match": {"patient": query_text}}}
    results = es.search(body = query, index = "patient_dev", size = 11, _source = False)
    result_ids = [x['_id'] for x in results['hits']['hits'][1:]]
    golden_list = dev_data[patient_id][0] + dev_data[patient_id][1]
    precision = getPrecision(golden_list, result_ids)
    recall = getRecall(golden_list, result_ids)
    NDCG = getNDCG(dev_data[patient_id][0], dev_data[patient_id][1], result_ids)
    precisions.append(precision)
    NDCGs.append(NDCG)
    recalls.append(recall)
    
    if precision == 0:
        results = es.search(body = query, index = "patient_dev", size = 10000, _source = False)
        result_ids = [x['_id'] for x in results['hits']['hits']]
        RR = getRR(golden_list, result_ids)
    else:
        RR = getRR(golden_list, result_ids)
    RRs.append(RR)

print("=========Dev=========")
print(np.mean(RRs), np.mean(precisions), np.mean(recalls), np.mean(NDCGs))
    

precisions = []
NDCGs = []
RRs = []
recalls = []
for patient_id in test_data:
    query_text = es.get(index = "patient_test", id = patient_id)['_source']['patient']
    if len(query_text) > 1024:
        query_text = query_text[:1024]
    query = {"query": {"match": {"patient": query_text}}}
    results = es.search(body = query, index = "patient_test", size = 11, _source = False)
    result_ids = [x['_id'] for x in results['hits']['hits'][1:]]
    golden_list = test_data[patient_id][0] + test_data[patient_id][1]
    precision = getPrecision(golden_list, result_ids)
    recall = getRecall(golden_list, result_ids)
    NDCG = getNDCG(test_data[patient_id][0], test_data[patient_id][1], result_ids)
    precisions.append(precision)
    NDCGs.append(NDCG)
    recalls.append(recall)
    
    if precision == 0:
        results = es.search(body = query, index = "patient_test", size = 10000, _source = False)
        result_ids = [x['_id'] for x in results['hits']['hits']]
        RR = getRR(golden_list, result_ids)
    else:
        RR = getRR(golden_list, result_ids)
    RRs.append(RR)

print("=========Test=========")
print(np.mean(RRs), np.mean(precisions), np.mean(recalls), np.mean(NDCGs))
    
