from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("..")
from threading import Thread, BoundedSemaphore
from queue import Queue
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader


data_dir = "../../../../../datasets/PAR"

es = ES(timeout = 1000)


class search_thread(Thread):
    def __init__(self, patient_id, patient):
        # 重写写父类的__init__方法
        super(search_thread, self).__init__()
        self.patient_id = patient_id
        self.patient = patient

    def run(self):
        query = {"query": {"multi_match": {"query": self.patient, "type": "cross_fields", "fields": ["title^3", "text"]}}}
        results = es.search(body = query, index = "par_corpus", size = 1000, _source = False)

        result_ids = {x['_id']: x['_score'] for x in results['hits']['hits']}
        
        q.put((self.patient_id, result_ids))
        thread_max.release()


thread_max = BoundedSemaphore(40)
q = Queue()
threads = []
result_ids_with_score = {}
corpus_path = "../../../../../datasets/PAR/corpus.jsonl"
query_path = "../../../../../datasets/queries/test_queries.jsonl"
qrels_path = "../../../../../datasets/PAR/qrels_test.tsv"
corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()

for patient_id in tqdm(queries):
    thread_max.acquire()
    t = search_thread(patient_id, queries[patient_id])
    t.start()
    threads.append(t)
for i in threads:
    i.join()

while not q.empty():
    result = q.get()
    result_ids_with_score[result[0]] = result[1]


print("=========Test=========")
evaluation = EvaluateRetrieval()
metrics = evaluation.evaluate(qrels, result_ids_with_score, [10, 1000])
print(metrics)
mrr = evaluation.evaluate_custom(qrels, result_ids_with_score, [len(corpus)], metric="mrr")
print(mrr)
json.dump(result_ids_with_score, open("../PAR_BM25_test.json", "w"), indent = 4)
