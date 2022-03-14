from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("..")
from utils import getNDCG, getPrecision, getRR, getRecall
from threading import Thread, BoundedSemaphore
from queue import Queue


data_dir = "../../../../../datasets/task_4_patient2article_retrieval"
test_data = json.load(open(os.path.join(data_dir, "PAR_test.json"), "r"))

es = ES(timeout = 1000)


class search_thread(Thread):
    def __init__(self, patient_id, rels, mode):
        # 重写写父类的__init__方法
        super(search_thread, self).__init__()
        self.patient_id = patient_id
        self.rels = rels
        self.mode = mode

    def run(self):
        query_text = es.get(index = "patient_" + self.mode, id = self.patient_id)['_source']['patient']
        query = {"query": {"multi_match": {"query": query_text, "type": "cross_fields", "fields": ["title^3", "abstract"]}}}
        results = es.search(body = query, index = "pubmed_title_abstract", size = 10000, _source = False)
        
        result_ids_with_score = [(x['_id'], x['_score']) for x in results['hits']['hits']]
        
        result_ids = [x['_id'] for x in results['hits']['hits'][:10]]
        golden_list = self.rels
        precision = getPrecision(golden_list, result_ids)

        result_ids = [x['_id'] for x in results['hits']['hits'][:1000]]
        recall_1k = getRecall(golden_list, result_ids)

        result_ids = [x['_id'] for x in results['hits']['hits']]
        RR = getRR(golden_list, result_ids)
        recall_10k = getRecall(golden_list, result_ids)
        
        q.put((RR, precision, recall_1k, recall_10k, self.patient_id, result_ids, result_ids_with_score))
        thread_max.release()


thread_max = BoundedSemaphore(40)
q = Queue()
precisions = []
recalls_1k = []
recalls_10k = []
RRs = []
threads = []
result_ids = {}
result_ids_with_score = {}

for patient_id in tqdm(test_data):
    thread_max.acquire()
    t = search_thread(patient_id, test_data[patient_id], "test")
    t.start()
    threads.append(t)
for i in threads:
    i.join()

while not q.empty():
    result = q.get()
    RRs.append(result[0])
    precisions.append(result[1])
    recalls_1k.append(result[2])
    recalls_10k.append(result[3])
    result_ids[result[4]] = result[5]
    result_ids_with_score[result[4]] = result[6]


print("=========Test=========")
print(np.mean(RRs), np.mean(precisions), np.mean(recalls_1k), np.mean(recalls_10k))
print(len(RRs))
json.dump(result_ids, open("../patient2article_retrieved_test.json", "w"), indent = 4)
json.dump(result_ids_with_score, open("../full_patient2article_retrieved_test.json", "w"), indent = 4)

