import json
import pandas as pd
import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np

# PMID2patient_uid dict
PMID2patients = {}
patients = json.load(open("../../../meta_data/PMC-Patients.json", "r"))

for patient in patients:
    if patient['PMID'] in PMID2patients.keys():
        PMID2patients[patient['PMID']].append(patient['patient_uid'])
    else:
        PMID2patients[patient['PMID']] = [patient['patient_uid']]

# result directory of extract_ref.py
citation_graph_dir = "../../../../pubmed/pubmed_citations"
file_list = os.listdir(citation_graph_dir)

cites = {}
cited_by = {} 

# Get articles that cite or are cited by any article.
# Note that there may be multiple files in pubmed archiving same article and the citations are taken union.
for file_name in tqdm(file_list):
    temp = json.load(open(os.path.join(citation_graph_dir, file_name)))
    for PMID in temp.keys():
        if PMID in PMID2patients.keys():
            if PMID not in cites:
                cites[PMID] = temp[PMID]
            else:
                cites[PMID] = list(set(temp[PMID]) | set(cites[PMID]))
        for cite_art in temp[PMID]:
            if cite_art in PMID2patients.keys():
                if cite_art in cited_by.keys():
                    cited_by[cite_art].append(PMID)
                else:
                    cited_by[cite_art] = [PMID]

# Not all articles have relevant articles.
# print(len(cites))
# print(len(cited_by))

# Some articles not in PubMed citaion graph have citations in PMC OA, take union of two.
PMC_cites = json.load(open("../../../../PMC_OA/PMC_cites.json"))

for PMID in tqdm(PMC_cites.keys()):
    if PMID in PMID2patients.keys():
        if PMID not in cites.keys():
            cites[PMID] = PMC_cites[PMID]
        else:
            cites[PMID] = list(set(PMC_cites[PMID]) | set(cites[PMID]))
    for cite_art in PMC_cites[PMID]:
        if cite_art in PMID2patients.keys():
            if cite_art in cited_by.keys():
                if PMID not in cited_by[cite_art]:
                    cited_by[cite_art].append(PMID)
            else:
                cited_by[cite_art] = [PMID]

# In case duplicates occur
cites = {k:list(set(v)) for k,v in cites.items()}
cited_by = {k:list(set(v)) for k,v in cited_by.items()}

# Remove the article itself in its cites and cited_by
for PMID in tqdm(PMID2patients.keys()):
    if PMID not in cites:
        cites[PMID] = []
    elif PMID in cites[PMID]:
        cites[PMID].remove(PMID)
    if PMID not in cited_by:
        cited_by[PMID] = []
    elif PMID in cited_by[PMID]:
        cited_by[PMID].remove(PMID)

# Article relevance dict, PMID2PMID.
relevant_article_PMID = {} 
# Articles from which PMC-Patients are extracted relevance dict.
relevant_article_with_note = {} 
for article in PMID2patients.keys():
    relevant_article_PMID[article] = list(set(cites[article] + cited_by[article]))
    relevant_article_with_note[article] = list(filter(lambda x: x in PMID2patients.keys(), relevant_article_PMID[article]))

"""
# Check symmetry of relevant annotation.
total = set()
for patient in relevant_article_with_note:
    for rel in relevant_article_with_note[patient]:
        total.add((patient, rel))

for x,y in total:
    if (y,x) not in total:
        print(x,y)
        import ipdb; ipdb.set_trace()
"""

# Define patient2patient similarity.
patient2patient_similarity = {}
for patient in tqdm(patients):
    PMID = patient['PMID']
    patient_id = patient['patient_uid']

    # First collect other notes from the same article, if any.
    sim_patient_id = deepcopy(PMID2patients[PMID])
    sim_patient_id.remove(patient_id)
    
    # Then collect notes extracted from relevant articles.
    sim_PMID = relevant_article_with_note[PMID]
    for sim in sim_PMID:
        for sim_patient in PMID2patients[sim]:
            sim_patient_id.append(sim_patient)

    patient2patient_similarity[patient_id] = sim_patient_id

"""
# Check symmetry of similarity annotation.
total = set()
for patient in patient2patient_similarity:
    for rel in patient2patient_similarity[patient]:
        total.add((patient, rel))

for x,y in total:
    if (y,x) not in total:
        import ipdb; ipdb.set_trace()
"""

json.dump(patient2patient_similarity, open("../../../meta_data/patient2patient_similarity.json", "w"), indent = 4)
del patient2patient_similarity


abstract_dir = "../../../../pubmed/pubmed_title_abstract"
PMIDs = set()
for file_name in tqdm(os.listdir(abstract_dir)):
    temp = json.load(open(os.path.join(abstract_dir, file_name), "r"))
    for PMID in temp.keys():
        PMIDs.add(PMID)
print(len(PMIDs))
json.dump(list(PMIDs), open("../../../datasets/patient2article_retrieval/PAR_PMIDs.json", "w"), indent = 4)

# All cites or cited by are defined to be relevant articles. 
# Note the article from which the patient note is extracted is included as well.
patient2article_relevance = {}
for patient in tqdm(patients):
    PMID = patient['PMID']
    patient_id = patient['patient_uid']
    patient2article_relevance[patient_id] = relevant_article_PMID[PMID] + [PMID]
    # Exclude relevant articles without abstract or title.
    patient2article_relevance[patient_id] = list(filter(lambda x: x in PMIDs, patient2article_relevance[patient_id]))

json.dump(patient2article_relevance, open("../../../meta_data/patient2article_relevance.json", "w"), indent = 4)

