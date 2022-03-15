import json
import os


human_patient_uids = json.load(open("../../../../meta_data/human_patient_uids.json", "r"))
retrieved = json.load(open("patient2article_retrieved_test_with_score.json", "r"))
patients = json.load(open("../../../../datasets/PMC-Patients_test.json", "r"))
patients = {x['patient_uid']: x for x in patients}
relevance = json.load(open("../../../../datasets/task_4_patient2article_retrieval/PAR_test.json", "r"))

pubmed_dir = "../../../../../pubmed/pubmed_title_abstract"
pubmeds = {}
for file_name in os.listdir(pubmed_dir):
    temp = json.load(open(os.path.join(pubmed_dir, file_name), "r"))
    for k in temp:
        temp[k]["PMID"] = k
    pubmeds.update(temp)

human_eval = []
for uid in human_patient_uids:
    patient = patients[uid]
    entry = {"query": patient}
    candidates = []
    for candidate in retrieved[uid][:10]:
        candidate_article = pubmeds[candidate[0]]
        candidate_article['score'] = candidate[1]
        if candidate[0] in relevance[uid]:
            candidate_article['label'] = 1
        else:
            candidate_article['label'] = 0
        candidates.append(candidate_article)
    entry['candidates'] = candidates
    human_eval.append(entry)


json.dump(human_eval, open("PAR_human_eval_top_10.json","w"), indent = 4)
