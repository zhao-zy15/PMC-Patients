from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("..")
from metrics import getPrecision, getRR, getRecall, getNDCG, getAP
from threading import Thread, BoundedSemaphore
from queue import Queue


data_dir = "../../../../../datasets/patient2patient_retrieval"

es = ES()

class search_thread(Thread):
    def __init__(self, patient_id, rels, mode):
        # 重写写父类的__init__方法
        super(search_thread, self).__init__()
        self.patient_id = patient_id
        self.rels = rels
        self.mode = mode

    def run(self):
        query_text = es.get(index = "patient_" + self.mode, id = self.patient_id)['_source']['patient']
        query = {"query": {"match": {"patient": {"query": query_text}}}}
        results = es.search(body = query, index = "patient_train", size = 1000, _source = False)

        result_ids_with_score = [(x['_id'], x['_score']) for x in results['hits']['hits']]
        
        result_ids = [x['_id'] for x in results['hits']['hits'][:10]]
        result_scores = [x['_score'] for x in results['hits']['hits'][:10]]
        golden_list = self.rels
        P = getPrecision(golden_list, result_ids)
        NDCG = getNDCG(golden_list, result_ids, result_scores)
        AP = getAP(golden_list, result_ids, result_scores)

        result_ids = [x['_id'] for x in results['hits']['hits']]
        R = getRecall(golden_list, result_ids)
        RR = getRR(golden_list, result_ids)

        q.put((RR, P, AP, NDCG, R, self.patient_id, result_ids_with_score))   
        thread_max.release()


thread_max = BoundedSemaphore(40)
q = Queue()
RR, P, AP, NDCG, R = ([],[],[],[],[])
threads = []
result_ids_with_score = {}
test_data = json.load(open(os.path.join(data_dir, "PPR_test.json"), "r"))

for patient_id in tqdm(test_data):
    thread_max.acquire()
    t = search_thread(patient_id, test_data[patient_id], "test")
    t.start()
    threads.append(t)
for i in threads:
    i.join()

while not q.empty():
    result = q.get()
    RR.append(result[0])
    P.append(result[1])
    AP.append(result[2])
    NDCG.append(result[3])
    R.append(result[4])
    result_ids_with_score[result[5]] = result[6]


print("=========Test=========")
print(np.mean(RR), np.mean(P), np.mean(AP), np.mean(NDCG), np.mean(R))
print(len(RR))
json.dump(result_ids_with_score, open("../PPR_BM25_test.json", "w"), indent = 4)
