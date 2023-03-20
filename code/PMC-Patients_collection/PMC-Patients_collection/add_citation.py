import json
import requests
from tqdm import tqdm
from threading import Thread, BoundedSemaphore
from queue import Queue


class MyThread(Thread):
    def __init__(self, url, patient_uid, PMCID):
        super(MyThread, self).__init__()
        self.url = url
        self.patient_uid = patient_uid
        self.PMCID = PMCID

    def run(self):
        params = {"format": "citation", "id": self.PMCID}
        response = requests.get(url = self.url, params = params, timeout = 100)
        citation_json = json.loads(response.text)
        citation = citation_json['nlm']['orig']
        q.put((self.patient_uid, citation))   
        thread_max.release()


patients = json.load(open("../../../meta_data/PMC-Patients.json", "r"))
base_url = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pmc/"

thread_max = BoundedSemaphore(45)
q = Queue()
threads = []
for patient in tqdm(patients):
    thread_max.acquire()
    PMCID = patient['file_path'].split('/')[-1][3 : -4]
    patient_uid = patient['patient_uid']
    t = MyThread(base_url, patient_uid, PMCID)
    t.start()
    threads.append(t)

for i in threads:
    i.join()

citations = {}
while not q.empty():
    result = q.get()
    citations[result[0]] = result[1]

json.dump(citations, open("../../../meta_data/PMC-Patients_citations.json", "w"), indent = 4)