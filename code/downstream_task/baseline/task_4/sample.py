import json
import numpy as np
from tqdm import tqdm
import  os


PMIDs = []
all_PMIDs = []
data_dir = "../../../../../pubmed/pubmed_title_abstract/"
for file_name in tqdm(os.listdir(data_dir)):
    articles = json.load(open(os.path.join(data_dir, file_name), "r"))
    PMIDs.append([])
    if '34748655' in articles.keys():
        import ipdb; ipdb.set_trace()
    PMIDs[-1] = list(filter(lambda x: len(articles[x]['title']) * len(articles[x]['abstract']) > 0, articles.keys()))
    if len(PMIDs[-1]) == 0:
        PMIDs.pop()
    all_PMIDs += PMIDs[-1]
all_PMIDs = set(all_PMIDs)

citations = {}
pubmed_citation_dir = "../../../../../pubmed/pubmed_citations"
for file_name in tqdm(os.listdir(pubmed_citation_dir)):
    cites = json.load(open(os.path.join(pubmed_citation_dir, file_name), "r"))
    citations.update(cites)


test = json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_test.json", "r"))
sample_test = {}
for patient_uid in tqdm(test):
    sample_test[patient_uid] = {}
    pos_candidates = test[patient_uid]
    pos_candidates = list(filter(lambda x: x in all_PMIDs, pos_candidates))
    sample_test[patient_uid]['pos'] = pos_candidates
    
    hard_neg_candidates = []
    for PMID in test[patient_uid]:
        if PMID in citations:
            hard_neg_candidates += citations[PMID]
    hard_neg_candidates = list(filter(lambda x: x in all_PMIDs, hard_neg_candidates))
    if len(hard_neg_candidates) <= 20:
        sample_test[patient_uid]['hard_neg'] = hard_neg_candidates
    else:
        sample_test[patient_uid]['hard_neg'] = np.random.choice(hard_neg_candidates, 10, replace = False).tolist()

    exclude = set(hard_neg_candidates + pos_candidates)
    neg_shards = np.random.choice(range(len(PMIDs)), 10)
    sample_test[patient_uid]['easy_neg'] = []
    for shard in neg_shards:
        easy_neg_candidates = np.random.choice(PMIDs[shard], 2).tolist()
        for candidate in easy_neg_candidates:
            if candidate not in exclude and candidate in all_PMIDs:
                sample_test[patient_uid]['easy_neg'].append(candidate)
json.dump(sample_test, open("../../../../datasets/task_4_patient2article_retrieval/PAR_sample/PAR_sample_test.json", "w"), indent = 4)


train = json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_train.json", "r"))
sample_train = {}
for patient_uid in tqdm(train):
    sample_train[patient_uid] = {}
    pos_candidates = train[patient_uid]
    pos_candidates = list(filter(lambda x: x in all_PMIDs, pos_candidates))
    if len(pos_candidates) <= 10:
        sample_train[patient_uid]['pos'] = pos_candidates
    else:
        sample_train[patient_uid]['pos'] = np.random.choice(pos_candidates, 10, replace = False).tolist()

    hard_neg_candidates = []
    for PMID in train[patient_uid]:
        if PMID in citations:
            hard_neg_candidates += citations[PMID]
    hard_neg_candidates = list(filter(lambda x: x in all_PMIDs, hard_neg_candidates))
    if len(hard_neg_candidates) <= 30:
        sample_train[patient_uid]['hard_neg'] = hard_neg_candidates
    else:
        sample_train[patient_uid]['hard_neg'] = np.random.choice(hard_neg_candidates, 10, replace = False).tolist()

    exclude = set(hard_neg_candidates + pos_candidates)
    neg_shards = np.random.choice(range(len(PMIDs)), 10)
    sample_train[patient_uid]['easy_neg'] = []
    for shard in neg_shards:
        easy_neg_candidates = np.random.choice(PMIDs[shard], 8).tolist()
        for candidate in easy_neg_candidates:
            if candidate not in exclude and candidate in all_PMIDs:
                sample_train[patient_uid]['easy_neg'].append(candidate)
json.dump(sample_train, open("../../../../datasets/task_4_patient2article_retrieval/PAR_sample/PAR_sample_train.json", "w"), indent = 4)


dev = json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_dev.json", "r"))
sample_dev = {}
for patient_uid in tqdm(dev):
    sample_dev[patient_uid] = {}
    pos_candidates = dev[patient_uid]
    pos_candidates = list(filter(lambda x: x in all_PMIDs, pos_candidates))
    sample_dev[patient_uid]['pos'] = pos_candidates

    hard_neg_candidates = []
    for PMID in dev[patient_uid]:
        if PMID in citations:
            hard_neg_candidates += citations[PMID]
    hard_neg_candidates = list(filter(lambda x: x in all_PMIDs, hard_neg_candidates))
    if len(hard_neg_candidates) <= 20:
        sample_dev[patient_uid]['hard_neg'] = hard_neg_candidates
    else:
        sample_dev[patient_uid]['hard_neg'] = np.random.choice(hard_neg_candidates, 10, replace = False).tolist()

    exclude = set(hard_neg_candidates + pos_candidates)
    neg_shards = np.random.choice(range(len(PMIDs)), 10)
    sample_dev[patient_uid]['easy_neg'] = []
    for shard in neg_shards:
        easy_neg_candidates = np.random.choice(PMIDs[shard], 2).tolist()
        for candidate in easy_neg_candidates:
            if candidate not in exclude and candidate in all_PMIDs:
                sample_dev[patient_uid]['easy_neg'].append(candidate)
json.dump(sample_dev, open("../../../../datasets/task_4_patient2article_retrieval/PAR_sample/PAR_sample_dev.json", "w"), indent = 4)