import json
import numpy as np
from tqdm import tqdm

np.random.seed(21)

PPR_train = json.load(open("../../../datasets/task_3_patient2patient_retrieval/PPR_train.json", "r"))
PPR_dev = json.load(open("../../../datasets/task_3_patient2patient_retrieval/PPR_dev.json", "r"))
PPR_test = json.load(open("../../../datasets/task_3_patient2patient_retrieval/PPR_test.json", "r"))

train_patient_uids = list(PPR_train.keys())
dev_patient_uids = list(PPR_dev.keys())
test_patient_uids = list(PPR_test.keys())

PPS_train = []
PPS_dev = []
PPS_test = []
# pairs in each split, used to drop duplicate.
train_pairs = set()
dev_pairs = set()
test_pairs = set()

for case_id in tqdm(PPR_train):
    ins_count = 0
    for rel_1 in PPR_train[case_id][0]:
        # When (x, y) is in instances, (y, x) shouldn't.
        if (rel_1, case_id) not in train_pairs:
            PPS_train.append((case_id, rel_1, 1))
            train_pairs.add((case_id, rel_1))
            ins_count += 1
    for rel_2 in PPR_train[case_id][1]:
        if (rel_2, case_id) not in train_pairs:
            PPS_train.append((case_id, rel_2, 2))
            train_pairs.add((case_id, rel_2))
            ins_count += 1
    # When sample negative samples, similar patients and itself should be excluded.
    rel = PPR_train[case_id][0] + PPR_train[case_id][1] + [case_id]
    candidate = list(filter(lambda x: x not in rel, train_patient_uids))
    for neg in np.random.choice(candidate, ins_count):
        PPS_train.append((case_id, neg, 0))
    train_pairs.add(case_id)

for case_id in tqdm(PPR_dev):
    ins_count = 0
    for rel_1 in PPR_dev[case_id][0]:
        if (rel_1, case_id) not in dev_pairs:
            PPS_dev.append((case_id, rel_1, 1))
            dev_pairs.add((case_id, rel_1))
            ins_count += 1
    for rel_2 in PPR_dev[case_id][1]:
        if (rel_2, case_id) not in dev_pairs:
            PPS_dev.append((case_id, rel_2, 2))
            dev_pairs.add((case_id, rel_2))
            ins_count += 1
    rel = PPR_dev[case_id][0] + PPR_dev[case_id][1] + [case_id]
    candidate = list(filter(lambda x: x not in rel, dev_patient_uids + train_patient_uids))
    for neg in np.random.choice(candidate, ins_count):
        PPS_dev.append((case_id, neg, 0))
    dev_pairs.add(case_id)

for case_id in tqdm(PPR_test):
    ins_count = 0
    for rel_1 in PPR_test[case_id][0]:
        if (rel_1, case_id) not in test_pairs:
            PPS_test.append((case_id, rel_1, 1))
            test_pairs.add((case_id, rel_1))
            ins_count += 1
    for rel_2 in PPR_test[case_id][1]:
        if (rel_2, case_id) not in test_pairs:
            PPS_test.append((case_id, rel_2, 2))
            test_pairs.add((case_id, rel_2))
            ins_count += 1
    rel = PPR_test[case_id][0] + PPR_test[case_id][1] + [case_id]
    candidate = list(filter(lambda x: x not in rel, test_patient_uids + train_patient_uids))
    for neg in np.random.choice(candidate, ins_count):
        PPS_test.append((case_id, neg, 0))
    test_pairs.add(case_id)

json.dump(PPS_train, open("../../../datasets/task_2_patient2patient_similarity/PPS_train.json", "w"), indent = 4)
json.dump(PPS_dev, open("../../../datasets/task_2_patient2patient_similarity/PPS_dev.json", "w"), indent = 4)
json.dump(PPS_test, open("../../../datasets/task_2_patient2patient_similarity/PPS_test.json", "w"), indent = 4)
