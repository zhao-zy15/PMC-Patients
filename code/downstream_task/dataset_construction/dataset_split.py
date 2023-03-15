import json
import numpy as np
import re
from tqdm import tqdm

# In case of dataset updates, sample and save dev and test PMIDs for dataset split in advance.
# New notes might be directly added into train set.
dev_PMIDs = set(json.load(open("../../../meta_data/dev_PMIDs.json", "r")))
test_PMIDs = set(json.load(open("../../../meta_data/test_PMIDs.json", "r")))
patients = json.load(open("../../../meta_data/PMC-Patients.json", "r"))

# Get patient_uids in dataset split and split PMC-Patients
train_patient_uids = set()
train_patients = []
dev_patient_uids = set()
dev_patients = []
test_patient_uids = set()
test_patients = []
for patient in tqdm(patients):
    if patient['PMID'] in dev_PMIDs:
        dev_patient_uids.add(patient['patient_uid'])
        dev_patients.append(patient)
    elif patient['PMID'] in test_PMIDs:
        test_patient_uids.add(patient['patient_uid'])
        test_patients.append(patient)
    else:
        train_patient_uids.add(patient['patient_uid'])
        train_patients.append(patient)

json.dump(train_patients, open("../../../datasets/PMC-Patients_train.json", "w"), indent = 4)
json.dump(dev_patients, open("../../../datasets/PMC-Patients_dev.json", "w"), indent = 4)
json.dump(test_patients, open("../../../datasets/PMC-Patients_test.json", "w"), indent = 4)

json.dump(list(train_patient_uids), open("../../../meta_data/train_patient_uids.json", "w"), indent = 4)
json.dump(list(dev_patient_uids), open("../../../meta_data/dev_patient_uids.json", "w"), indent = 4)
json.dump(list(test_patient_uids), open("../../../meta_data/test_patient_uids.json", "w"), indent = 4)

# Split patient2patient retrieval (PPR) dataset
relevant_patient2patient = json.load(open("../../../meta_data/patient2patient_similarity.json", "r"))

PPR_train = {}
PPR_dev = {}
PPR_test = {}
for patient_uid in tqdm(relevant_patient2patient.keys()):
    # Only train_patient_uids are used as documents.
    if patient_uid in train_patient_uids:
        PPR_train[patient_uid] = []
        for rel_case in relevant_patient2patient[patient_uid]:
            if rel_case in train_patient_uids:
                PPR_train[patient_uid].append(rel_case)
        # Remove queries without relevant patients.
        if len(PPR_train[patient_uid]) == 0:
            del PPR_train[patient_uid]
            
    if patient_uid in dev_patient_uids:
        PPR_dev[patient_uid] = []
        for rel_case in relevant_patient2patient[patient_uid]:
            if rel_case in train_patient_uids:
                PPR_dev[patient_uid].append(rel_case)
        if len(PPR_dev[patient_uid]) == 0:
            del PPR_dev[patient_uid]
            
    if patient_uid in test_patient_uids:
        PPR_test[patient_uid] = []
        for rel_case in relevant_patient2patient[patient_uid]:
            if rel_case in train_patient_uids:
                PPR_test[patient_uid].append(rel_case)
        if len(PPR_test[patient_uid]) == 0:
            del PPR_test[patient_uid]

json.dump(PPR_train, open("../../../datasets/patient2patient_retrieval/PPR_train.json", "w"), indent = 4)
json.dump(PPR_dev, open("../../../datasets/patient2patient_retrieval/PPR_dev.json", "w"), indent = 4)
json.dump(PPR_test, open("../../../datasets/patient2patient_retrieval/PPR_test.json", "w"), indent = 4)

# Split patient2article (PAR) dataset.
relevant_patient2article = json.load(open("../../../meta_data/patient2article_relevance.json", "r"))
PAR_train = {}
PAR_dev = {}
PAR_test = {}
for patient_uid in tqdm(relevant_patient2article.keys()):
    if patient_uid in train_patient_uids:
        PAR_train[patient_uid] = relevant_patient2article[patient_uid]
    if patient_uid in dev_patient_uids:
        PAR_dev[patient_uid] = relevant_patient2article[patient_uid]
    if patient_uid in test_patient_uids:
        PAR_test[patient_uid] = relevant_patient2article[patient_uid]

json.dump(PAR_train, open("../../../datasets/task_4_patient2article_retrieval/PAR_train.json", "w"), indent = 4)
json.dump(PAR_dev, open("../../../datasets/task_4_patient2article_retrieval/PAR_dev.json", "w"), indent = 4)
json.dump(PAR_test, open("../../../datasets/task_4_patient2article_retrieval/PAR_test.json", "w"), indent = 4)
