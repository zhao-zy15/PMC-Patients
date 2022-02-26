import json
from tqdm import tqdm
import numpy as np

np.random.seed(21)
# Construct citation graph.
PPS = json.load(open("../../../meta_data/patient2patient_similarity.json", "r"))
PPS = {k: v[0] + v[1] for k, v in PPS.items()}
PPS_graph = {}
non_easy = {}
for patient in tqdm(PPS):
    # Entry i of the list is the list of patients at hop i.
    # Hop 0 are positive samples.
    PPS_graph[patient] = [[], [], [], [], [], []]
    # Hop <= 5 are considered as non-easy-negatives.
    # Including positive samples and hard negatives (may including false negatives).
    non_easy[patient] = set([patient])
    # Using BFS to search for patients at hop i.
    queue = [(patient, -1)]
    while queue:
        temp = queue.pop(0)
        for next_p in PPS[temp[0]]:
            if next_p not in non_easy[patient] and temp[1] <= 4:
                queue.append((next_p, temp[1] + 1))
                PPS_graph[patient][temp[1] + 1].append(next_p)
                non_easy[patient].add(next_p)

json.dump(PPS_graph, open("../../../meta_data/PPS_graph.json", "w"), indent = 4)


patient_uids_train = set(json.load(open("../../../meta_data/train_patient_uids.json", "r")))
patient_uids_dev = set(json.load(open("../../../meta_data/dev_patient_uids.json", "r")))
patient_uids_test = set(json.load(open("../../../meta_data/test_patient_uids.json", "r")))

PPS_train = []
PPS_dev = []
PPS_test = []
"""
    Add positive samples into PPS dataset.
"""
# Used to drop duplicates.
added = {uid: set([]) for uid in patient_uids_train}
for patient in tqdm(patient_uids_train):
    positives = PPS_graph[patient][0]
    for pos in positives:
        if pos in added[patient]:
            continue
        # PPS train set only contains pairs of patients both from PMC-Patients-train.
        if pos in patient_uids_dev or pos in patient_uids_test:
            continue
        # Same article, similarity 2.
        if pos.split('-')[0] == patient.split('-')[0]:
            PPS_train.append((patient, pos, 2))
        else:
            PPS_train.append((patient, pos, 1))
        added[patient].add(pos)
        added[pos].add(patient)
    

added = {uid: set([]) for uid in patient_uids_dev}
for patient in tqdm(patient_uids_dev):
    positives = PPS_graph[patient][0]
    for pos in positives:
        # PPS dev contains train+dev or dev+dev.
        if pos in patient_uids_test:
            continue
        if pos in patient_uids_dev:
            if pos in added[patient]:
                continue
            else:
                if pos.split('-')[0] == patient.split('-')[0]:
                    PPS_dev.append((patient, pos, 2))
                else:
                    PPS_dev.append((patient, pos, 1))
                added[patient].add(pos)
                added[pos].add(patient)
        if pos in patient_uids_train:
            PPS_dev.append((patient, pos, 1))
        

added = {uid: set([]) for uid in patient_uids_test}
for patient in tqdm(patient_uids_test):
    positives = PPS_graph[patient][0]
    for pos in positives:
        # PPS test contains train+test or test+test.
        if pos in patient_uids_dev:
            continue
        if pos in patient_uids_test:
            if pos in added[patient]:
                continue
            else:
                if pos.split('-')[0] == patient.split('-')[0]:
                    PPS_test.append((patient, pos, 2))
                else:
                    PPS_test.append((patient, pos, 1))
                added[patient].add(pos)
                added[pos].add(patient)
        if pos in patient_uids_train:
            PPS_test.append((patient, pos, 1))
        

"""
    Sample hard negatives (5 or 6-hop patients).
"""
added = {uid: set([]) for uid in patient_uids_train}
for patient in tqdm(patient_uids_train):
    # Ensure that there are no more hard negative than positive samples for each patient.
    remain_num = len(PPS_graph[patient][0]) - len(added[patient])
    if remain_num <= 0:
        continue
    candidates = list(filter(lambda x: x in patient_uids_train and x not in added[patient], PPS_graph[patient][4] + PPS_graph[patient][5]))
    hard_neg = np.random.choice(candidates, min(remain_num, len(candidates)), replace = False)
    for neg in hard_neg:
        PPS_train.append((patient, neg, 0))
        added[patient].add(neg)
        added[neg].add(patient)
    
added = {uid: set([]) for uid in patient_uids_dev}
for patient in tqdm(patient_uids_dev):
    remain_num = len(PPS_graph[patient][0]) - len(added[patient])
    if remain_num <= 0:
        continue
    candidates = list(filter(lambda x: x not in patient_uids_test and x not in added[patient], PPS_graph[patient][4] + PPS_graph[patient][5]))
    hard_neg = np.random.choice(candidates, min(remain_num, len(candidates)), replace = False)
    for neg in hard_neg:
        PPS_dev.append((patient, neg, 0))
        if neg in patient_uids_dev:
            added[patient].add(neg)
            added[neg].add(patient)

added = {uid: set([]) for uid in patient_uids_test}
for patient in tqdm(patient_uids_test):
    remain_num = len(PPS_graph[patient][0]) - len(added[patient])
    if remain_num <= 0:
        continue
    candidates = list(filter(lambda x: x not in patient_uids_dev and x not in added[patient], PPS_graph[patient][4] + PPS_graph[patient][5]))
    hard_neg = np.random.choice(candidates, min(remain_num, len(candidates)), replace = False)
    for neg in hard_neg:
        PPS_test.append((patient, neg, 0))
        if neg in patient_uids_test:
            added[patient].add(neg)
            added[neg].add(patient)
    

"""
    Sample easy negatives (outside 6-hop patients).
    Sample 2 easy negatives for each patient.
"""
easy_neg_num = 2
added = {uid: set([]) for uid in patient_uids_train}
for patient in tqdm(patient_uids_train):
    remain_num = easy_neg_num - len(added[patient])
    if remain_num <= 0:
        continue
    candidates = list(filter(lambda x: x not in non_easy[patient] and x not in added[patient], patient_uids_train))
    easy_neg = np.random.choice(candidates, remain_num, replace = False)
    for neg in easy_neg:
        PPS_train.append((patient, neg, -1))
        added[patient].add(neg)
        added[neg].add(patient)


added = {uid: set([]) for uid in patient_uids_dev}
for patient in tqdm(patient_uids_dev):
    remain_num = easy_neg_num - len(added[patient])
    if remain_num <= 0:
        continue
    candidates = list(filter(lambda x: x not in non_easy[patient] and x not in added[patient], patient_uids_train | patient_uids_dev))
    easy_neg = np.random.choice(candidates, remain_num, replace = False)
    for neg in easy_neg:
        PPS_dev.append((patient, neg, -1))
        if neg in patient_uids_dev:
            added[patient].add(neg)
            added[neg].add(patient)


added = {uid: set([]) for uid in patient_uids_test}
for patient in tqdm(patient_uids_test):
    remain_num = easy_neg_num - len(added[patient])
    if remain_num <= 0:
        continue
    candidates = list(filter(lambda x: x not in non_easy[patient] and x not in added[patient], patient_uids_train | patient_uids_test))
    easy_neg = np.random.choice(candidates, remain_num, replace = False)
    for neg in easy_neg:
        PPS_test.append((patient, neg, -1))
        if neg in patient_uids_test:
            added[patient].add(neg)
            added[neg].add(patient)


"""
    Check 1. id_1 != id_2; 2. dataset split; 3. no replicate.
"""
def check():
    added = set()
    for pair in PPS_train:
        assert pair[0] != pair[1]
        assert pair[0] in patient_uids_train and pair[1] in patient_uids_train
        assert (pair[0], pair[1]) not in added and (pair[1], pair[0]) not in added
        added.add((pair[0], pair[1]))
    for pair in PPS_dev:
        assert pair[0] != pair[1]
        assert pair[0] in patient_uids_dev
        assert pair[1] not in patient_uids_test
        assert (pair[0], pair[1]) not in added and (pair[1], pair[0]) not in added
        added.add((pair[0], pair[1]))
    for pair in PPS_test:
        assert pair[0] != pair[1]
        assert pair[0] in patient_uids_test
        assert pair[1] not in patient_uids_dev
        assert (pair[0], pair[1]) not in added and (pair[1], pair[0]) not in added
        added.add((pair[0], pair[1]))
    print("End check.")

check()

json.dump(PPS_train, open("../../datasets/task_2_patient2patient_similarity/PPS_train.json", "w"), indent = 4)
json.dump(PPS_dev, open("../../datasets/task_2_patient2patient_similarity/PPS_dev.json", "w"), indent = 4)
json.dump(PPS_test, open("../../datasets/task_2_patient2patient_similarity/PPS_test.json", "w"), indent = 4)
