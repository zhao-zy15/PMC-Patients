import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


patients = json.load(open("../../meta_data/PMC-Patients.json", "r"))
# PMID2patient_uid dict
PMID2patient = {}
for patient in patients:
    if patient['PMID'] in PMID2patient.keys():
        PMID2patient[patient['PMID']].append(patient['patient_uid'])
    else:
        PMID2patient[patient['PMID']] = [patient['patient_uid']]

# Number of articles from which patients are extracted.
print(len(PMID2patient))
# Number of patient notes.
print(len(patients))
# Word count of each patient note.
length = [len(x) for x in PMID2patient.values()]
print("Avg patient per note: ", np.mean(length))

patient_finding = json.load(open("../../../downstream_task/meta_data/patient_note_recognition.json", "r"))
para_count = 0
for patient in patient_finding:
    for tag in patient['tags']:
        if tag != 'O':
            para_count += 1
print("Avg para per patient: ", para_count / len(patients))

length = [len(x['patient'].split('. ')) for x in patients]
print("Avg sent per patient: ", np.mean(length))

length = [len(x['patient'].split()) for x in patients]
print("Avg word per patient: ", np.mean(length))

# Length distribution
print("Length greater than 3k: ", len(list(filter(lambda x: x > 3000, length))))
length = list(filter(lambda x: x <= 3000, length))
data = pd.DataFrame({"id":range(len(length)),"len":length})
groups = data.groupby(["len"]).count()
plt.bar(groups.index.to_list(), groups['id'].to_list())
plt.savefig("../../figures/word_counts_distribution.png")
plt.clf()

# Demographic distribution
male_count = 0
ages = []
for patient in patients:
    if patient['gender'] == "M":
        male_count += 1
    if patient['age'][0][1] != "year":
        ages.append(0)
    else:
        ages.append(int(np.floor(patient['age'][0][0])))
print("Percent of males: {:.5f}".format(male_count / len(patients)))
print("Average age: {:.3f}".format(np.mean(ages)))
data = pd.DataFrame({"id":range(len(ages)),"len":ages})
groups = data.groupby(["len"]).count()
plt.bar(groups.index.to_list(), groups['id'].to_list())
plt.savefig("../../figures/age_distribution.png")
