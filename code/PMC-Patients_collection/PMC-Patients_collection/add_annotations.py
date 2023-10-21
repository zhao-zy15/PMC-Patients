import json


patients = json.load(open("../../../datasets/PMC-Patients.json", "r"))
PAR = json.load(open("../../../meta_data/patient2article_relevance.json", "r"))
PPR = json.load(open("../../../meta_data/patient2patient_similarity.json", "r"))
PMIDs = set([patient['PMID'] for patient in patients])

new_data = []
for patient in patients:
    new_data.append(patient)
    pmcid = patient['patient_uid'].split('-')[0]
    new_data[-1]['relevant_articles'] = {PMID: 2 if PMID in PMIDs else 1 for PMID in PAR[patient['patient_uid']]}
    new_data[-1]['similar_patients'] = {patient_uid: 2 if patient_uid.split('-')[0] == pmcid else 1 for patient_uid in PPR[patient['patient_uid']]}

json.dump(new_data, open("../../../datasets/PMC-Patients.json", "w"), indent = 4)
