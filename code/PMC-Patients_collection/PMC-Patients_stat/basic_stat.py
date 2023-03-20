import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


patients = json.load(open("../../../meta_data/PMC-Patients.json", "r"))
# PMID2patient_uid dict
PMID2patient = {}
for patient in patients:
    if patient['PMID'] in PMID2patient.keys():
        PMID2patient[patient['PMID']].append(patient['patient_uid'])
    else:
        PMID2patient[patient['PMID']] = [patient['patient_uid']]

# Number of articles from which patients are extracted.
print("Article count: ", len(PMID2patient))
# Number of patient notes.
print("Patient count: ", len(patients))
# Word count of each patient note.
length = [len(x) for x in PMID2patient.values()]
print("Avg patient per article: ", np.mean(length))

length = [len(re.split(r'\.[ \n]', x['patient'])) for x in patients]
print("Avg sent per patient: ", np.mean(length))

length = []
for patient in patients:
    text = patient['patient'].lower()
    text = re.sub(r"[^a-z]", ' ', text)
    text = re.sub(r" +", ' ', text)
    length.append(len(list(filter(lambda x: len(x) > 0, text.split()))))

print("Avg word per patient: ", np.mean(length))
import ipdb; ipdb.set_trace()