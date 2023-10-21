import json
import jsonlines
import numpy as np
import re
from tqdm import tqdm

# In case of dataset updates, sample and save dev and test PMIDs for dataset split in advance.
# New notes might be directly added into train set.
dev_PMIDs = set(json.load(open("../../../meta_data/dev_PMIDs.json", "r")))
test_PMIDs = set(json.load(open("../../../meta_data/test_PMIDs.json", "r")))
PMIDs = set(json.load(open("../../../meta_data/PMIDs.json", "r")))

patients = json.load(open("../../../datasets/PMC-Patients.json", "r"))
PPR_corpus = []
train_queries = []
dev_queries = []
test_queries = []

# Get patient_uids in dataset split and split PMC-Patients
train_patient_uids = set()
dev_patient_uids = set()
test_patient_uids = set()
for patient in tqdm(patients):
    if patient['PMID'] in dev_PMIDs:
        dev_patient_uids.add(patient['patient_uid'])
        dev_queries.append({"_id": patient['patient_uid'], "text": patient['patient']})
    elif patient['PMID'] in test_PMIDs:
        test_patient_uids.add(patient['patient_uid'])
        test_queries.append({"_id": patient['patient_uid'], "text": patient['patient']})
    else:
        train_patient_uids.add(patient['patient_uid'])
        train_queries.append({"_id": patient['patient_uid'], "text": patient['patient']})
        PPR_corpus.append({"_id": patient['patient_uid'], "title": "", "text": patient['patient']})

# json.dump(list(train_patient_uids), open("../../../meta_data/train_patient_uids.json", "w"), indent = 4)
# json.dump(list(dev_patient_uids), open("../../../meta_data/dev_patient_uids.json", "w"), indent = 4)
# json.dump(list(test_patient_uids), open("../../../meta_data/test_patient_uids.json", "w"), indent = 4)

# with jsonlines.open('../../../datasets/patient2patient_retrieval/PPR_corpus.jsonl', mode='w') as writer:
#     for doc in PPR_corpus:
#         writer.write(doc)
        
# with jsonlines.open('../../../datasets/queries/train_queries.jsonl', mode='w') as writer:
#     for query in train_queries:
#         writer.write(query)
        
# with jsonlines.open('../../../datasets/queries/dev_queries.jsonl', mode='w') as writer:
#     for query in dev_queries:
#         writer.write(query)
        
# with jsonlines.open('../../../datasets/queries/test_queries.jsonl', mode='w') as writer:
#     for query in test_queries:
#         writer.write(query)

# Split patient2article (PAR) dataset.
relevant_patient2article = json.load(open("../../../meta_data/patient2article_relevance.json", "r"))
PAR_train = {}
PAR_dev = {}
PAR_test = {}
for patient_uid in tqdm(relevant_patient2article.keys()):
    if patient_uid in train_patient_uids:
        PAR_train[patient_uid] = {rel: 2 if rel in PMIDs else 1 for rel in relevant_patient2article[patient_uid]}
    if patient_uid in dev_patient_uids:
        PAR_dev[patient_uid] = {rel: 2 if rel in PMIDs else 1 for rel in relevant_patient2article[patient_uid]}
    if patient_uid in test_patient_uids:
        PAR_test[patient_uid] = {rel: 2 if rel in PMIDs else 1 for rel in relevant_patient2article[patient_uid]}

for dataset_type, dataset in [("train", PAR_train), ("dev", PAR_dev), ("test", PAR_test)]:
    with open(f"../../../datasets/PAR/qrels_{dataset_type}.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for patient_uid, rel_dict in dataset.items():
            for rel, score in rel_dict.items():
                f.write(f"{patient_uid}\t{rel}\t{score}\n")


# Split patient2patient retrieval (PPR) dataset
relevant_patient2patient = json.load(open("../../../meta_data/patient2patient_similarity.json", "r"))

PPR_train = {}
PPR_dev = {}
PPR_test = {}
for patient_uid in tqdm(relevant_patient2patient.keys()):
    # Only train_patient_uids are used as documents.
    if patient_uid in train_patient_uids:
        PPR_train[patient_uid] = {}
        for rel_case in relevant_patient2patient[patient_uid]:
            if rel_case in train_patient_uids:
                if rel_case.split('-')[0] == patient_uid.split('-')[0]:
                    PPR_train[patient_uid][rel_case] = 2
                else:
                    PPR_train[patient_uid][rel_case] = 1
        # Remove queries without relevant patients.
        if len(PPR_train[patient_uid]) == 0:
            del PPR_train[patient_uid]
            
    if patient_uid in dev_patient_uids:
        PPR_dev[patient_uid] = {}
        for rel_case in relevant_patient2patient[patient_uid]:
            if rel_case in train_patient_uids:
                PPR_dev[patient_uid][rel_case] = 1
        if len(PPR_dev[patient_uid]) == 0:
            del PPR_dev[patient_uid]
            
    if patient_uid in test_patient_uids:
        PPR_test[patient_uid] = {}
        for rel_case in relevant_patient2patient[patient_uid]:
            if rel_case in train_patient_uids:
                PPR_test[patient_uid][rel_case] = 1
        if len(PPR_test[patient_uid]) == 0:
            del PPR_test[patient_uid]


for dataset_type, dataset in [("train", PPR_train), ("dev", PPR_dev), ("test", PPR_test)]:
    with open(f"../../../datasets/PPR/qrels_{dataset_type}.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for patient_uid, rel_dict in dataset.items():
            for rel, score in rel_dict.items():
                f.write(f"{patient_uid}\t{rel}\t{score}\n")
