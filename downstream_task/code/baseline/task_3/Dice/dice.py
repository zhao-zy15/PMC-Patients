import json
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import getRR, getPrecision, getRecall


def dice(set_1, set_2):
    return 2 * len(set_1 & set_2) / (len(set_1) + len(set_2))


uid2NER = json.load(open("../../task_2/feature_based/NER.json", "r"))
data = json.load(open("../../../../../datasets/task_3_patient2patient_retrieval/PPR_test.json", "r"))
patient_ids = json.load(open("../../../../../meta_data/test_patient_uids.json", "r"))
patient_ids += json.load(open("../../../../../meta_data/train_patient_uids.json", "r"))
print(len(patient_ids))


RR = []
precision = []
recall_1000 = []
recall = []
for patient in tqdm(data):
    set_1 = set(uid2NER[patient])
    temp = []
    for candidate in patient_ids:
        if candidate == patient:
            continue
        temp.append((candidate, dice(set_1, set(uid2NER[candidate]))))
    temp = sorted(temp, key = lambda x:x[1], reverse = True)
    result_ids = [x[0] for x in temp]
    golden_labels = data[patient][0] + data[patient][1]
    RR.append(getRR(golden_labels, result_ids))
    precision.append(getPrecision(golden_labels, result_ids[:10]))
    recall_1000.append(getRecall(golden_labels, result_ids[:1000]))
    recall.append(getRecall(golden_labels, result_ids[:10000]))

print(np.mean(RR), np.mean(precision), np.mean(recall_1000), np.mean(recall))
print(len(RR))