import json
from elasticsearch import Elasticsearch as ES
from tqdm import tqdm
from threading import Thread, BoundedSemaphore
from queue import Queue
import os


es = ES(timeout = 1000)

class search_thread(Thread):
    def __init__(self, patient):
        # 重写写父类的__init__方法
        super(search_thread, self).__init__()
        self.patient = patient
        self.query_text = patient['patient']

    def run(self):
        query = {"query": {"multi_match": {"query": self.query_text, "type": "cross_fields", "fields": ["title^3", "abstract"]}}}

        results = es.search(body = query, index = "pubmed_title_abstract", size = 10, _source = True)
        results = [(x['_id'], x['_score'], x['_source']) for x in results['hits']['hits']]

        q.put((self.patient, results))   
        thread_max.release()


patients_human = json.load(open("../../../../datasets/PMC-Patients_human.json", "r"))
patients = json.load(open("../../../../datasets/PMC-Patients_test.json", "r"))
patients = {x['patient_uid']: x for x in patients}
relevance = json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_test.json", "r"))


thread_max = BoundedSemaphore(40)
q = Queue()
threads = []
for patient in tqdm(patients_human):
    thread_max.acquire()
    t = search_thread(patient)
    t.start()
    threads.append(t)
for i in threads:
    i.join()


human_eval = []
while not q.empty():
    result = q.get()
    patient = result[0]
    PMCID = patient['human_patient_uid'].split("-")[0]
    temp_uid = PMCID + '-1'
    ES_results = result[1]
    entry = {"query": patient}
    candidates = []
    for candidate in ES_results:
        candidate_article = {"PMID": candidate[0]}
        candidate_article.update(candidate[2])
        candidate_article['score'] = candidate[1]
        if candidate[0] in relevance[temp_uid]:
            candidate_article['label'] = 1
        else:
            candidate_article['label'] = 0
        candidates.append(candidate_article)
    entry['candidates'] = candidates
    human_eval.append(entry)


json.dump(human_eval, open("PAR_human_eval_top_10.json","w"), indent = 4)



