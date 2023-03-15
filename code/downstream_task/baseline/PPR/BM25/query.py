from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("..")
from utils import getPrecision, getRR, getRecall
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
        results = es.search(body = query, index = "patient_train", size = 10000, _source = False)

        result_ids_with_score = [(x['_id'], x['_score']) for x in results['hits']['hits']]
        
        result_ids = [x['_id'] for x in results['hits']['hits'][:10]]
        golden_list = self.rels
        p_10 = getPrecision(golden_list, result_ids)
        p_5 = getPrecision(golden_list, result_ids[:5])
        p_3 = getPrecision(golden_list, result_ids[:3])
        p_1 = getPrecision(golden_list, result_ids[:1])

        result_ids = [x['_id'] for x in results['hits']['hits'][:1000]]
        recall_1k = getRecall(golden_list, result_ids)

        result_ids = [x['_id'] for x in results['hits']['hits']]
        RR = getRR(golden_list, result_ids)
        recall_10k = getRecall(golden_list, result_ids)

        q.put((RR, p_1, p_3, p_5, p_10, recall_1k, recall_10k, self.patient_id, result_ids, result_ids_with_score))   
        thread_max.release()


thread_max = BoundedSemaphore(40)
q = Queue()
RR, p_1, p_3, p_5, p_10, recall_1k, recall_10k = ([],[],[],[],[],[],[])
threads = []
result_ids = {}
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
    p_1.append(result[1])
    p_3.append(result[2])
    p_5.append(result[3])
    p_10.append(result[4])
    recall_1k.append(result[5])
    recall_10k.append(result[6])
    result_ids[result[7]] = result[8]
    result_ids_with_score[result[7]] = result[9]


print("=========Test=========")
print(np.mean(RR), np.mean(p_1), np.mean(p_3), np.mean(p_5), np.mean(p_10), np.mean(recall_1k), np.mean(recall_10k))
print(len(RR))
json.dump(result_ids_with_score, open("../BM25_results_test.json", "w"), indent = 4)
