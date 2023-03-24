import os
import json
import jsonlines
from tqdm import tqdm

PAR = json.load(open("../../../meta_data/patient2article_relevance.json", "r"))
PAR_PMIDs = set()
for patient in PAR:
    for PMID in PAR[patient]:
        PAR_PMIDs.add(PMID)

human = json.load(open("../../../meta_data/relation_human_annotations.json", "r"))
for patient in human:
    for idx in human[patient]:
        if '-' not in idx and idx not in PAR_PMIDs:
            PAR_PMIDs.add(idx)

print(len(PAR_PMIDs))
json.dump(list(PAR_PMIDs), open("../../../meta_data/PAR_PMIDs.json", "w"), indent = 4)

abstract_dir = "../../../../pubmed/pubmed_title_abstract"
PAR_corpus = []
written = set()
for file_name in tqdm(os.listdir(abstract_dir)):
    temp = json.load(open(os.path.join(abstract_dir, file_name), "r"))
    for PMID in temp:
        if PMID in PAR_PMIDs and PMID not in written:
            PAR_corpus.append({"_id": PMID, "title": temp[PMID]["title"], "text": temp[PMID]["abstract"]})
            written.add(PMID)

print(len(PAR_corpus))
with jsonlines.open('../../../datasets/patient2article_retrieval/PAR_corpus.jsonl', mode='w') as writer:
    for doc in PAR_corpus:
        writer.write(doc)
