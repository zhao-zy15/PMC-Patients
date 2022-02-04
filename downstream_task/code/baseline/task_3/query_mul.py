from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm
import numpy as np
from utils import getNDCG, getPrecision, getRR, getRecall
from threading import Thread, BoundedSemaphore
from queue import Queue


data_dir = "../../../datasets/task_3_patient2patient_retrieval"
train_data = json.load(open(os.path.join(data_dir, "patient2patient_train.json"), "r"))
dev_data = json.load(open(os.path.join(data_dir, "patient2patient_dev.json"), "r"))
test_data = json.load(open(os.path.join(data_dir, "patient2patient_test.json"), "r"))

es = ES(http_auth = "elastic:UpMvQdoGeuoHDfrv00qF")

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
        results = es.search(body = query, index = "patient_" + self.mode, size = 11, _source = False)
        result_ids = [x['_id'] for x in results['hits']['hits'][1:]]
        golden_list = self.rels[0] + self.rels[1]
        precision = getPrecision(golden_list, result_ids)
        #NDCG = getNDCG(self.rels[0], self.rels[1], result_ids)

        results = es.search(body = query, index = "patient_" + self.mode, size = 1001, _source = False)
        result_ids = [x['_id'] for x in results['hits']['hits'][1:]]
        recall_1k = getRecall(golden_list, result_ids)

        results = es.search(body = query, index = "patient_" + self.mode, size = 10001, _source = False)
        result_ids = [x['_id'] for x in results['hits']['hits'][1:]]
        RR = getRR(golden_list, result_ids)
        recall_10k = getRecall(golden_list, result_ids)

        q.put((RR, precision, recall_1k, recall_10k))
        thread_max.release()


thread_max = BoundedSemaphore(40)

q = Queue()
precisions = []
recalls_1k = []
recalls_10k = []
#NDCGs = []
RRs = []
threads = []
for patient_id in tqdm(dev_data):
    thread_max.acquire()
    t = search_thread(patient_id, dev_data[patient_id], "dev")
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
    #NDCGs.append(result[3])

print("=========Dev=========")
print(np.mean(RRs), np.mean(precisions), np.mean(recalls_1k), np.mean(recalls_10k))
print(len(RRs))


q = Queue()
precisions = []
recalls_1k = []
recalls_10k = []
#NDCGs = []
RRs = []
threads = []
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
    #NDCGs.append(result[3])

print("=========Test=========")
print(np.mean(RRs), np.mean(precisions), np.mean(recalls_1k), np.mean(recalls_10k))
print(len(RRs))