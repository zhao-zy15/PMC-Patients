import json
import numpy as np


train = json.load(open("../../../datasets/patient2patient_retrieval/PPR_train.json", "r"))
sim = []
for patient_id in train:
    sim.append(len(train[patient_id]))
print("===Train===")
print("Similarity: ", np.sum(sim))
print("Avg similarity: ", np.sum(sim) / len(sim))
print("Queries: ", len(train))

dev = json.load(open("../../../datasets/patient2patient_retrieval/PPR_dev.json", "r"))
sim = []
for patient_id in dev:
    sim.append(len(dev[patient_id]))
print("===Dev===")
print("Similarity: ", np.sum(sim))
print("Avg similarity: ", np.sum(sim) / len(sim))
print("Queries: ", len(dev))

test = json.load(open("../../../datasets/patient2patient_retrieval/PPR_test.json", "r"))
sim = []
for patient_id in test:
    sim.append(len(test[patient_id]))
print("===Test===")
print("Similarity: ", np.sum(sim))
print("Avg similarity: ", np.sum(sim) / len(sim))
print("Queries: ", len(test))