import json


human_patient_uids = json.load(open("../../../../meta_data/human_patient_uids.json", "r"))
retrieved = json.load(open("full_patient2patient_retrieved_test.json", "r"))
patients = json.load(open("../../../../meta_data/PMC-Patients.json", "r"))
patients = {x['patient_uid']: x for x in patients}
similarity = json.load(open("../../../../datasets/task_3_patient2patient_retrieval/PPR_test.json", "r"))

human_eval = []
for uid in human_patient_uids:
    patient = patients[uid]
    entry = {"query": patient}
    candidates = []
    for candidate in retrieved[uid][:10]:
        candidate_patient = patients[candidate[0]]
        candidate_patient['score'] = candidate[1]
        if uid not in similarity:
            candidate_patient['label'] = 0
        elif candidate[0] in similarity[uid][0]:
            candidate_patient['label'] = 1
        elif candidate[0] in similarity[uid][1]:
            candidate_patient['label'] = 2
        else:
            candidate_patient['label'] = 0
        candidates.append(candidate_patient)
    entry['candidates'] = candidates
    human_eval.append(entry)

json.dump(human_eval, open("PPR_human_eval_top_10.json","w"), indent = 4)
import ipdb; ipdb.set_trace()