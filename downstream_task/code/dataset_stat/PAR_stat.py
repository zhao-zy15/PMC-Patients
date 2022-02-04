import json
import numpy as np
import os
from tqdm import tqdm

# Document collection: all pubmed titles and abstracts
abstract_dir = "../../../../pubmed/pubmed_title_abstract"
PMIDs = set()
for file_name in tqdm(os.listdir(abstract_dir)):
    temp = json.load(open(os.path.join(abstract_dir, file_name), "r"))
    for PMID in temp.keys():
        PMIDs.add(PMID)
print(len(PMIDs))

PAR_train = json.load(open("../../datasets/task_4_patient2article_retrieval/PAR_train.json", "r"))
relevant_count = []
for patient_id in PAR_train:
    relevant_count.append(len(PAR_train[patient_id]))
print("Labels count: ", np.sum(relevant_count))
print("Average labels per note", np.mean(relevant_count))
print("Queries: ", len(PAR_train))

PAR_dev = json.load(open("../../datasets/task_4_patient2article_retrieval/PAR_dev.json", "r"))
relevant_count = []
for patient_id in PAR_dev:
    relevant_count.append(len(PAR_dev[patient_id]))
print("Labels count: ", np.sum(relevant_count))
print("Average labels per note", np.mean(relevant_count))
print("Queries: ", len(PAR_dev))


PAR_test = json.load(open("../../datasets/task_4_patient2article_retrieval/PAR_test.json", "r"))
relevant_count = []
for patient_id in PAR_test:
    relevant_count.append(len(PAR_test[patient_id]))
print("Labels count: ", np.sum(relevant_count))
print("Average labels per note", np.mean(relevant_count))
print("Queries: ", len(PAR_test))
