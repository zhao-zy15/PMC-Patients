import json
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import getRR, getPrecision, getRecall
import numpy as np

PPR = json.load(open("../../task_3/full_patient2patient_retrieved_test.json"))
train_data = json.load(open("../../../../../datasets/task_4_patient2article_retrieval/PAR_train.json", "r"))
test_data = json.load(open("../../../../../datasets/task_4_patient2article_retrieval/PAR_test.json", "r"))

# k = 0, take until candidates > 1W.
k = 0
RR = []
precision = []
recall_1k = []
recall = []
for patient in tqdm(test_data):
    candidate = {}
    for patient_id, score in PPR[patient][:k]:
        if patient_id in train_data:
            for rel in train_data[patient_id]:
                if rel in candidate:
                    candidate[rel] += score
                else:
                    candidate[rel] = score
    if len(candidate) < 10000:
        cur = k
        while len(candidate) < 10000:
            patient_id, score = PPR[patient][cur]
            if patient_id in train_data:
                for rel in train_data[patient_id]:
                    if rel in candidate:
                        candidate[rel] += score
                    else:
                        candidate[rel] = score
            cur += 1

    golden_labels = test_data[patient]
    pred_labels = sorted(list(candidate.items()), key = lambda x: x[1], reverse = True)
    pred_labels = [x[0] for x in pred_labels]

    RR.append(getRR(golden_labels, pred_labels))
    precision.append(getPrecision(golden_labels, pred_labels[:10]))
    recall_1k.append(getRecall(golden_labels, pred_labels[:1000]))
    recall.append(getRecall(golden_labels, pred_labels[:10000]))

print(np.mean(RR), np.mean(precision), np.mean(recall_1k), np.mean(recall))
print(k)