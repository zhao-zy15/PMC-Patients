import json
import numpy as np


train = json.load(open("../../../datasets/task_3_patient2patient_retrieval/PPR_train.json", "r"))
sim_1 = []
sim_2 = []
for patient_id in train:
    sim_1.append(len(train[patient_id][0]))
    sim_2.append(len(train[patient_id][1]))
print("Similarity 1 labels: ", np.sum(sim_1))
print("Similarity 2 labels: ", np.sum(sim_2))
print("Similarity in total: ", np.sum(sim_1) + np.sum(sim_2))
print("Avg similarity 1 labels per query: ", np.mean(sim_1))
print("Avg similarity 2 labels per query: ", np.mean(sim_2))
print("Avg similarity in total: ", (np.sum(sim_1) + np.sum(sim_2)) / len(sim_1))
print("Queries: ", len(train))

dev = json.load(open("../../../datasets/task_3_patient2patient_retrieval/PPR_dev.json", "r"))
sim_1 = []
sim_2 = []
for patient_id in dev:
    sim_1.append(len(dev[patient_id][0]))
    sim_2.append(len(dev[patient_id][1]))
print("Similarity 1 labels: ", np.sum(sim_1))
print("Similarity 2 labels: ", np.sum(sim_2))
print("Similarity in total: ", np.sum(sim_1) + np.sum(sim_2))
print("Avg similarity 1 labels per query: ", np.mean(sim_1))
print("Avg similarity 2 labels per query: ", np.mean(sim_2))
print("Avg similarity in total: ", (np.sum(sim_1) + np.sum(sim_2)) / len(sim_1))
print("Queries: ", len(dev))

test = json.load(open("../../../datasets/task_3_patient2patient_retrieval/PPR_test.json", "r"))
sim_1 = []
sim_2 = []
for patient_id in test:
    sim_1.append(len(test[patient_id][0]))
    sim_2.append(len(test[patient_id][1]))
print("Similarity 1 labels: ", np.sum(sim_1))
print("Similarity 2 labels: ", np.sum(sim_2))
print("Similarity in total: ", np.sum(sim_1) + np.sum(sim_2))
print("Avg similarity 1 labels per query: ", np.mean(sim_1))
print("Avg similarity 2 labels per query: ", np.mean(sim_2))
print("Avg similarity in total: ", (np.sum(sim_1) + np.sum(sim_2)) / len(sim_1))
print("Queries: ", len(test))