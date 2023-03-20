import os
import json
from tqdm import tqdm

PAR = json.load(open("../../../meta_data/patient2article_relevance.json", "r"))
PAR_PMIDs = set()
for patient in PAR:
    for PMID in PAR[patient]:
        PAR_PMIDs.add(PMID)

human = json.load(open("../../../datasets/patient2article_retrieval/PPR_PAR_human_annotations.json", "r"))
for patient in human:
    for idx in human[patient]:
        if '-' not in idx and idx not in PAR_PMIDs:
            PAR_PMIDs.add(idx)

print(len(PAR_PMIDs))
json.dump(list(PAR_PMIDs), open("../../../datasets/patient2article_retrieval/PAR_PMIDs.json", "w"), indent = 4)

abstract_dir = "../../../../pubmed/pubmed_title_abstract"
PAR_abstracts = {}
for file_name in tqdm(os.listdir(abstract_dir)):
    temp = json.load(open(os.path.join(abstract_dir, file_name), "r"))
    for PMID in temp:
        if PMID in PAR_PMIDs:
            PAR_abstracts[PMID] = temp[PMID]

print(len(PAR_abstracts))
json.dump(PAR_abstracts, open("../../../datasets/patient2article_retrieval/PAR_abstracts.json", "w"), indent = 4)
