import json
import numpy as np 
from scipy.stats.mstats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


human = json.load(open("../../../datasets/patient2patient_retrieval/PPR_PAR_human_annotations.json", "r"))
label = json.load(open("../baseline/PPR/PPR_human_eval_top_10.json", "r"))

scores = [[], []]
for ins in label:
    patient_id = ins['query']['human_patient_uid']
    for i in range(5):
        candidate = ins['candidates'][i]
        patient_uid = candidate['patient_uid']
        ground_truth = human[patient_id][patient_uid]
        if ground_truth == '0':
            score = 0
        else:
            score = len(ground_truth)
        if candidate['label'] == 0:
            scores[0].append(score)
        else:
            scores[1].append(score)

print(np.mean(scores[0]), np.mean(scores[1]))
y0 = np.bincount(scores[0])
y0_ppr = y0 / sum(y0) * 100
y1 = np.bincount(scores[1])
y1_ppr = y1 / sum(y1) * 100
print(ttest_ind(scores[0], scores[1], equal_var = False))

label = json.load(open("../baseline/PAR/PAR_human_eval_top_10.json", "r"))
scores = [[], []]
for ins in label:
    patient_id = ins['query']['human_patient_uid']
    for i in range(5):
        candidate = ins['candidates'][i]
        PMID = candidate['PMID']
        ground_truth = human[patient_id][PMID]
        if ground_truth == '0':
            score = 0
        else:
            score = len(ground_truth)
        if candidate['label'] == 0:
            scores[0].append(score)
        else:
            scores[1].append(score)

print(np.mean(scores[0]), np.mean(scores[1]))
y0 = np.bincount(scores[0])
y0_par = y0 / sum(y0) * 100
y1 = np.bincount(scores[1])
y1_par = y1 / sum(y1) * 100
print(ttest_ind(scores[0], scores[1], equal_var = False))

plt.rcParams['figure.figsize'] = (8.5,2)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

ax2 = plt.axes([.06, 0.215, .4, .78])
ax1 = plt.axes([.56, 0.215, .4, .78])


bar_width = 0.3
colors = ["firebrick", "steelblue"]
ax1.bar(np.array([0,1,2,3]) - bar_width / 2, y0_ppr, width = bar_width, color = colors[0], label = 'Dissimilar in PMC-P')
ax1.bar(np.array([0,1,2,3]) + bar_width / 2, y1_ppr, width = bar_width, color = colors[1], label = 'Similar in PMC-P')
ax1.set_xlabel('Human similarity score', fontsize=12)
ax1.set_ylabel('Percent of pairs', fontsize=12)
ax1.set_xticks([0,1,2,3])
ax1.set_xticklabels(['0', '1', '2', '3'])
ax1.legend(loc = 'upper left')

ax2.bar(np.array([0,1,2,3]) - bar_width / 2, y0_par, width = bar_width, color = colors[0], label = 'Irrelevant in PMC-P')
ax2.bar(np.array([0,1,2,3]) + bar_width / 2, y1_par, width = bar_width, color = colors[1], label = 'Relevant in PMC-P')
ax2.set_xlabel('Human relevance score', fontsize=12)
ax2.set_ylabel('Percent of pairs', fontsize=12)
ax2.set_xticks([0,1,2,3])
ax2.set_xticklabels(['0', '1', '2', '3'])
ax2.legend(loc = 'upper left')


plt.savefig("../../../figures/relation_annotation_quality.pdf", format = 'pdf')
