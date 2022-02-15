import faiss
import numpy as np 
import os
import json
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import getRR, getPrecision, getRecall


embeddings = np.load("embeddings.npy")
embeddings = embeddings / np.linalg.norm(embeddings, axis = 1).reshape(-1, 1)
embed_row2id = []
embed_row2uid = []

patients = json.load(open("../../../../datasets/PMC-Patients_test.json", "r"))
for patient in patients:
    embed_row2id.append(int(patient['patient_id']))
    embed_row2uid.append(patient['patient_uid'])
queries = embeddings[embed_row2id, :]
patients = json.load(open("../../../../datasets/PMC-Patients_train.json", "r"))
for patient in patients:
    embed_row2id.append(int(patient['patient_id']))
    embed_row2uid.append(patient['patient_uid'])
documents = embeddings[embed_row2id, :]

dim = 768

nlist = 1024
k = 10001
m =  24
quantizer = faiss.IndexFlatIP(dim)
#index = faiss.IndexFlatIP(dim)
#index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
#index = faiss.index_factory(dim, "PCA256, IVF1024, PQ8")

print(index.is_trained)
index.train(documents)
print(index.is_trained)
index.add(documents)
index.nprobe = 100
print(index.ntotal)  


data = json.load(open("../../../../datasets/task_3_patient2patient_retrieval/PPR_test.json", "r"))

print("Begin search...")
results = index.search(queries, k)
print("End search!")

RR, precision, recall_1k, recall_10k = ([],[],[],[])
for result in tqdm(results[1]):
    result_ids = [embed_row2uid[hit] for hit in result]
    if result_ids[0] in data:
        golden_list = data[result_ids[0]][0] + data[result_ids[0]][1]
        precision.append(getPrecision(golden_list, result_ids[1:11]))
        recall_1k.append(getRecall(golden_list, result_ids[1:1001]))
        RR.append(getRR(golden_list, result_ids[1:]))
        recall_10k.append(getRecall(golden_list, result_ids[1:]))

print(np.mean(RR), np.mean(precision), np.mean(recall_1k), np.mean(recall_10k))
print(len(RR))
