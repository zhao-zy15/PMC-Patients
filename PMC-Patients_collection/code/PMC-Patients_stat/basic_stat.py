import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


patients = json.load(open("../../meta_data/PMC-Patients.json", "r"))
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

patient_finding = json.load(open("../../../downstream_task/meta_data/patient_note_recognition.json", "r"))
para_count = 0
for patient in patient_finding:
    for tag in patient['tags']:
        if tag != 'O':
            para_count += 1
print("Avg para per patient: ", para_count / len(patients))

length = [len(re.split(r'\.[ \n]', x['patient'])) for x in patients]
print("Avg sent per patient: ", np.mean(length))

length = [len(x['patient'].split()) for x in patients]
print("Avg word per patient: ", np.mean(length))

# Length distribution
print("Length greater than 2500: ", len(list(filter(lambda x: x > 2500, length))))
length = list(filter(lambda x: x <= 2500, length))
#np.save("PMC_word_length.npy", np.array(length))
data = pd.DataFrame({"id":range(len(length)),"len":length})
groups = data.groupby(["len"]).count()

plt.rcParams['figure.figsize'] = (3.7,2.8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.xlabel('Length in word count', fontsize=12)
plt.ylabel('Relative frequency of notes', fontsize=12)
plt.tick_params(labelsize=8)
plt.grid(linestyle=':', axis='y')
plt.bar(groups.index.to_list(), (groups['id'] / sum(groups['id'])).to_list(), color='steelblue', align='center')
plt.axvline(x = np.mean(length), ymin = 0, ymax = 0.9, ls = '--', lw = 0.75, color = 'darkred')
plt.annotate(text = "Avg. {:.2f}".format(np.mean(length)), xy = (420, 0.002), \
    xytext = (700, 0.0018), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))
plt.tight_layout(pad = 0.01)
plt.savefig("../../figures/PMC-Patients_length_dist.pdf", format = "pdf")
plt.clf()

# Demographic distribution
male_count = 0
ages = []
infant_ages = []
for patient in patients:
    if patient['gender'] == "M":
        male_count += 1

    age_value = 0
    for value, unit in patient['age']:
        if unit == 'year':
            age_value += value
        if unit == 'month':
            age_value += value / 12
        if unit == 'week':
            age_value += value / 52.14
        if unit == 'day':
            age_value += value / 365
        if unit == 'hour':
            age_value += value / 365 / 24
    if age_value < 1:
        infant_ages.append(age_value)
    ages.append(int(np.floor(age_value)))

print("Percent of males: {:.5f}".format(male_count / len(patients)))
print("Average age: {:.3f}".format(np.mean(ages)))
data = pd.DataFrame({"id":range(len(ages)),"len":ages})
groups = data.groupby(["len"]).count()

plt.rcParams['figure.figsize'] = (3.7,2.8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.xlabel('Age', fontsize=12)
plt.ylabel('Relative frequency of notes', fontsize=12)
plt.tick_params(labelsize=8)
plt.xticks(ticks = np.arange(0, 101, 10), labels = np.arange(0, 101, 10))
plt.grid(linestyle=':', axis='y')
plt.bar(groups.index.to_list(), (groups['id'] / sum(groups['id'])).to_list(), color='steelblue', align='center')
plt.axvline(x = np.mean(ages), ymin = 0, ymax = 0.95, ls = '--', lw = 0.75, color = 'darkred')
plt.annotate(text = "Avg. {:.2f}".format(np.mean(ages)), xy = (43, 0.0175), \
    xytext = (5, 0.017), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))
plt.tight_layout(pad = 0.01)
plt.savefig("../../figures/PMC-Patients_age_dist.png", format = "png")
plt.clf()

data = pd.DataFrame({"id":range(len(infant_ages)),"len":infant_ages})
groups = data.groupby(["len"]).count()

print("Avg. infant ages: ", np.mean(infant_ages))
#import ipdb; ipdb.set_trace()
plt.rcParams['figure.figsize'] = (3.7,2.8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.xlabel('Age', fontsize=12)
plt.ylabel('Relative frequency of notes', fontsize=12)
plt.tick_params(labelsize=8)
#plt.xticks(ticks = np.arange(0, 101, 10), labels = np.arange(0, 101, 10))
plt.grid(linestyle=':', axis='y')
plt.bar(groups.index.to_list(), (groups['id'] / sum(groups['id'])).to_list(), 0.01, color='steelblue', align='center')
#plt.axvline(x = np.mean(ages), ymin = 0, ymax = 0.95, ls = '--', lw = 0.75, color = 'darkred')
#plt.annotate(text = "Avg. {:.2f}".format(np.mean(ages)), xy = (43, 0.0175), \
#    xytext = (5, 0.017), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))
plt.tight_layout(pad = 0.01)
plt.savefig("../../figures/PMC-Patients_infant_age_dist.png", format = "png")
plt.clf()
