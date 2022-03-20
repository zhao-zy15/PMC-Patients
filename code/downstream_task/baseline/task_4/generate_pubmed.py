import json
from tqdm import tqdm
import os


PAR_data = json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_train.json", "r"))
PAR_data.update(json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_dev.json", "r")))
PAR_data.update(json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_test.json", "r")))
print(len(PAR_data))

PMIDs = set()
for candidate in tqdm(PAR_data.values()):
    for PMID in candidate:
        PMIDs.add(PMID)

print(len(PMIDs))

pubmed_PAR = {}
data_dir = "../../../../../pubmed/pubmed_title_abstract/"
for file_name in tqdm(os.listdir(data_dir)):
    articles = json.load(open(os.path.join(data_dir, file_name), "r"))
    for PMID in articles:
        if PMID in PMIDs:
            pubmed_PAR[PMID] = articles[PMID]

print(len(pubmed_PAR))

json.dump(pubmed_PAR, open("pubmed_PAR.json", "w"), indent = 4)