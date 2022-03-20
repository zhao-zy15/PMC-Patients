import json
import numpy as np
from tqdm import tqdm
import  os
from elasticsearch import Elasticsearch as ES



citations = {}
pubmed_citation_dir = "../../../../../pubmed/pubmed_citations"
for file_name in tqdm(os.listdir(pubmed_citation_dir)):
    cites = json.load(open(os.path.join(pubmed_citation_dir, file_name), "r"))
    citations.update(cites)


es = ES()
train = json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_train.json", "r"))
sample_train = {}
for patient_uid in tqdm(train):
    sample_train[patient_uid] = {}
    query_text = es.get(index = "patient_train", id = patient_uid)['_source']['patient']
    query = {"query": {"multi_match": {"query": query_text, "type": "cross_fields", "fields": ["abstract"]}}}
    pos_candidates = train[patient_uid]
    scores = []
    if len(pos_candidates) > 50:
        pos_candidates = np.random.choice(pos_candidates, 50, replace = False)
    for candidate in pos_candidates:
        scores.append(es.explain(body = query, index = "pubmed_title_abstract", id = candidate)['explanation']['value'])
    scores = np.array(scores)
    sample_train[patient_uid]['pos'] = np.random.choice(pos_candidates, min(10, len(pos_candidates)), \
        replace = False, p = np.log(scores + 1) / sum(np.log(scores + 1))).tolist()

    hard_neg_candidates_full = []
    for PMID in train[patient_uid]:
        if PMID in citations:
            hard_neg_candidates_full += citations[PMID]
    if len(pos_candidates) > 50:
        hard_neg_candidates = np.random.choice(hard_neg_candidates_full, 50, replace = False)
    for candidate in hard_neg_candidates:
        scores.append(es.explain(body = query, index = "pubmed_title_abstract", id = candidate)['explanation']['value'])
    scores = np.array(scores)
    sample_train[patient_uid]['hard_neg'] = np.random.choice(hard_neg_candidates, min(10, len(hard_neg_candidates)), \
        replace = False, p = np.log(scores + 1) / sum(np.log(scores + 1))).tolist()
    


    import ipdb; ipdb.set_trace()
    
