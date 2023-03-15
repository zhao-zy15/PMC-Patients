import json
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import getRR, getPrecision, getRecall
import numpy as np

PPR = json.load(open("../../PPR/full_patient2patient_retrieved_test.json"))
train_data = json.load(open("../../../../../datasets/atient2article_retrieval/PAR_train.json", "r"))
test_data = json.load(open("../../../../../datasets/patient2article_retrieval/PAR_test.json", "r"))

# k = 0, take until candidates > 1W.
k = 0
RR, p_1, p_3, p_5, p_10, recall_1k, recall_10k = ([],[],[],[],[],[],[])
results = {}
results_with_score = {}
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
    results_with_score[patient] = pred_labels
    pred_labels = [x[0] for x in pred_labels]
    results[patient] = pred_labels
    RR.append(getRR(golden_labels, pred_labels))
    p_1.append(getPrecision(golden_labels, pred_labels[:1]))
    p_3.append(getPrecision(golden_labels, pred_labels[:3]))
    p_5.append(getPrecision(golden_labels, pred_labels[:5]))
    p_10.append(getPrecision(golden_labels, pred_labels[:10]))
    recall_1k.append(getRecall(golden_labels, pred_labels[:1000]))
    recall_10k.append(getRecall(golden_labels, pred_labels[:10000]))

print(np.mean(RR), np.mean(p_1), np.mean(p_3), np.mean(p_5), np.mean(p_10), np.mean(recall_1k), np.mean(recall_10k))
print(k)