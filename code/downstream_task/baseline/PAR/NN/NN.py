import json
from tqdm import tqdm
import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval


#retrieved = json.load(open("../../PPR/PPR_Dense_test_full.json", "r"))
retrieved = json.load(open("../../PPR/PPR_BM25_test_full.json", "r"))
for query in retrieved:
    retrieved[query] = sorted(retrieved[query].items(), key = lambda x: x[1], reverse = True)

train_data = {}
with open("../../../../../datasets/PAR/qrels_train.tsv", "r") as f:
    lines = f.readlines()
for line in lines[1:]:
    q, doc, _ = line.split('\t')
    if q in train_data:
        train_data[q].append(doc)
    else:
        train_data[q] = [doc]
qrels = {}
with open("../../../../../datasets/PAR/qrels_test.tsv", "r") as f:
    lines = f.readlines()
for line in lines[1:]:
    q, doc, score = line.split('\t')
    score = "1"
    if q in qrels:
        qrels[q][doc] = int(score)
    else:
        qrels[q] = {doc: int(score)}

# k = 0, take until candidates > 1W.
k = 0
results = {}
for patient in tqdm(qrels):
    candidate = {}
    for patient_id, score in retrieved[patient][:k]:
        if patient_id in candidate:
            candidate[patient_id] += score
        else:
            candidate[patient_id] = score
        if patient_id in train_data:
            for sim in train_data[patient_id]:
                if sim in candidate:
                    candidate[sim] += score
                else:
                    candidate[sim] = score
    if len(candidate) < 10000:
        cur = k
        while len(candidate) < 10000:
            patient_id, score = retrieved[patient][cur]
            if patient_id in candidate:
                candidate[patient_id] += score
            else:
                candidate[patient_id] = score
            if patient_id in train_data:
                for sim in train_data[patient_id]:
                    if sim in candidate:
                        candidate[sim] += score
                    else:
                        candidate[sim] = score
            cur += 1

    pred_labels = sorted(list(candidate.items()), key = lambda x: x[1], reverse = True)
    results[patient] = {x[0]: x[1] for x in pred_labels}

evaluation = EvaluateRetrieval()
metrics = evaluation.evaluate(qrels, results, [10, 1000])
mrr = evaluation.evaluate_custom(qrels, results, [10000], metric="mrr")
print(mrr[f'MRR@{10000}'], metrics[3]['P@10'], metrics[0]['NDCG@10'], metrics[2]['Recall@1000'])