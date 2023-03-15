import json
import sys
from tqdm import tqdm
sys.path.append("..")
from utils import getRR, getPrecision, getRecall
import numpy as np

BM25_results = json.load(open("BM25_results_dev.json", "r"))
patients = json.load(open("../../../../../meta_data/PMC-Patients.json", "r"))
data = json.load(open("../../../../../datasets/patient2patient_retrieval/PPR_dev.json", "r"))

uid2age = {}
# Convert ages into float numbers.
for patient in patients:
    uid = patient['patient_uid']
    age = 0
    for (value, unit) in patient['age']:
        if unit == "year":
            age += value
        if unit == "month":
            age += value / 12
        if unit == "week":
            age += value / 52
        if unit == "day":
            age += value / 365
        if unit == "hour":
            age += value / 365 / 24
    uid2age[uid] = age

alpha = float(sys.argv[1])

for query in tqdm(BM25_results):
    for candidate in BM25_results[query]:
        candidate_id = candidate[0]
        age_diff = min(uid2age[query], uid2age[candidate_id]) / max(uid2age[query], uid2age[candidate_id])
        new_score = float(candidate[1]) + alpha * age_diff
        candidate[1] = new_score

RR = []
precision = []
recall_1000 = []
recall = []
for patient in tqdm(data):
    temp = sorted(BM25_results[patient], key = lambda x:x[1], reverse = True)
    result_ids = [x[0] for x in temp]
    golden_labels = data[patient]
    RR.append(getRR(golden_labels, result_ids))
    precision.append(getPrecision(golden_labels, result_ids[:5]))
    recall_1000.append(getRecall(golden_labels, result_ids[:1000]))
    recall.append(getRecall(golden_labels, result_ids[:10000]))

print(np.mean(RR), np.mean(precision), np.mean(recall_1000), np.mean(recall))
print(alpha)