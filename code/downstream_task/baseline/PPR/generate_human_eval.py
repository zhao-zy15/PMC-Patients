import json
from elasticsearch import Elasticsearch as ES
from tqdm import tqdm
from threading import Thread, BoundedSemaphore
from queue import Queue


es = ES()

class search_thread(Thread):
    def __init__(self, patient):
        # 重写写父类的__init__方法
        super(search_thread, self).__init__()
        self.patient = patient
        self.query_text = patient['patient']

    def run(self):
        query = {"query": {"match": {"patient": {"query": self.query_text}}}}

        results = es.search(body = query, index = "patient_test", size = 11, _source = True)
        top1 = results['hits']['hits'][0]['_source']['patient']
        if top1 in self.query_text or self.query_text in top1:
            result_ids_with_score = [(x['_id'], x['_score']) for x in results['hits']['hits'][1:]]
        else:
            result_ids_with_score = [(x['_id'], x['_score']) for x in results['hits']['hits'][:-1]]

        q.put((self.patient, result_ids_with_score))   
        thread_max.release()


patients_human = json.load(open("../../../../datasets/PMC-Patients_human.json", "r"))
patients = json.load(open("../../../../datasets/PMC-Patients_test.json", "r"))
patients += json.load(open("../../../../datasets/PMC-Patients_train.json", "r"))
patients = {x['patient_uid']: x for x in patients}
similarity = json.load(open("../../../../datasets/task_3_patient2patient_retrieval/PPR_test.json", "r"))


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
    result_ids = result[1]
    entry = {"query": patient}
    candidates = []
    for candidate in result_ids:
        candidate_patient = patients[candidate[0]]
        candidate_patient['score'] = candidate[1]
        
        if temp_uid not in similarity:
            candidate_patient['label'] = 0
        elif candidate[0] in similarity[temp_uid][0]:
            candidate_patient['label'] = 1
        elif candidate[0].split('-')[0] == PMCID:
            candidate_patient['label'] = 2
        else:
            candidate_patient['label'] = 0
        candidates.append(candidate_patient)
    entry['candidates'] = candidates
    human_eval.append(entry)


json.dump(human_eval, open("PPR_human_eval_top_10.json","w"), indent = 4)
