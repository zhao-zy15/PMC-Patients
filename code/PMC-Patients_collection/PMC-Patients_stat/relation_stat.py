import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


patients = json.load(open("../../../datasets/PMC-Patients.json", "r"))
relevant_article = {p['patient_uid']: p["relevant_articles"] for p in patients}

print("====== Article Relevance =======")
print("Article count: ", len(relevant_article))
length_1 = [len([k for k, v in arts.items() if v == 1]) for arts in relevant_article.values()]
length_2 = [len([k for k, v in arts.items() if v == 2]) for arts in relevant_article.values()]
print("-------- Relevance 1 ---------")
print("Relevance count: ", sum(length_1))
print("Average Relevance per article: ", np.mean(length_1))
print("Length greater than 50: ", len(list(filter(lambda x: x > 50, length_1))))
print("-------- Relevance 2 ---------")
print("Relevance count: ", sum(length_2))
print("Average Relevance per article: ", np.mean(length_2))
print("Length greater than 50: ", len(list(filter(lambda x: x > 50, length_2))))


mean_length_1 = np.mean(length_1)
length_1 = list(filter(lambda x: x <= 50, length_1))
data_1 = pd.DataFrame({"id":range(len(length_1)),"len":length_1})
groups_1 = data_1.groupby(["len"]).count()

mean_length_2 = np.mean(length_2)
length_2 = list(filter(lambda x: x <= 50, length_2))
data_2 = pd.DataFrame({"id":range(len(length_2)),"len":length_2})
groups_2 = data_2.groupby(["len"]).count()


plt.rcParams['figure.figsize'] = (8.5,2)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

ax1 = plt.axes([.05, 0.215, .4, .78])
ax2 = plt.axes([.56, 0.215, .4, .523])
ax3 = plt.axes([.56, 0.784, .4, .211])
color = [["steelblue", "turquoise"], ["firebrick", "pink"]]

ax1.set_xlabel('(a) Number of relevance annotations', fontsize=12)
ax1.set_ylabel('Percent of patients', fontsize=12)
ax1.tick_params(labelsize=10)
ax1.set_xticks(np.arange(0, 101, 10))
ax1.set_xticklabels(np.arange(0, 101, 10))
ax1.grid(linestyle=':', axis='y')
bar_width = 0.4
ax1.bar((groups_2.index - bar_width * 0.5).to_list(), (groups_2['id'] / sum(groups_2['id']) * 100).to_list(), bar_width, color=color[1][0], align='center', label = "Relevance 2")
ax1.axvline(x = mean_length_2, ymin = 0, ymax = 0.85, ls = '--', lw = 1.5, color = color[1][1])
ax1.annotate(text = "Avg: {:.1f}".format(mean_length_2), xy = (2.5, 40), \
    xytext = (7, 43), fontsize = 12, arrowprops = dict(arrowstyle = '->', color = 'black'))
ax1.bar((groups_1.index + bar_width * 0.5).to_list(), (groups_1['id'] / sum(groups_1['id']) * 100).to_list(), bar_width, color=color[0][0], align='center', label = "Relevance 1")
ax1.axvline(x = mean_length_1, ymin = 0, ymax = 0.2, ls = '--', lw = 1.5, color = color[0][1])
ax1.annotate(text = "Avg: {:.1f}".format(mean_length_1), xy = (17, 6.1), \
    xytext = (21.5, 8.6), fontsize = 12, arrowprops = dict(arrowstyle = '->', color = 'black'))
ax1.legend(loc = 'upper right')


patient_similarity = {p['patient_uid']: p["similar_patients"] for p in patients}

print("====== Patient Similarity=======")
length_1 = [len([k for k, v in arts.items() if v == 1]) for arts in patient_similarity.values()]
length_2 = [len([k for k, v in arts.items() if v == 2]) for arts in patient_similarity.values()]
print("-------- Similarity 1 ---------")
print("Similarity count: ", sum(length_1))
print("Average Similarity per article: ", np.mean(length_1))
print("Length greater than 10: ", len(list(filter(lambda x: x > 10, length_1))))
print("-------- Similarity 2 ---------")
print("Similarity count: ", sum(length_2))
print("Average Similarity per article: ", np.mean(length_2))
print("Length greater than 10: ", len(list(filter(lambda x: x > 10, length_2))))

mean_length_1 = np.mean(length_1)
length_1 = list(filter(lambda x: x <= 10, length_1))
data_1 = pd.DataFrame({"id":range(len(length_1)),"len":length_1})
groups_1 = data_1.groupby(["len"]).count()

mean_length_2 = np.mean(length_2)
length_2 = list(filter(lambda x: x <= 10, length_2))
data_2 = pd.DataFrame({"id":range(len(length_2)),"len":length_2})
groups_2 = data_2.groupby(["len"]).count()


color = [["steelblue", "turquoise"], ["firebrick", "pink"]]
bar_width = 0.4
ax2.set_xlabel('(b) Number of similarity annotations', fontsize=12)
ax2.set_ylabel('Percent of patients', fontsize=12, y = 0.746)
ax2.tick_params(labelsize=10)
ax3.tick_params(labelbottom = False, bottom = False, labelsize = 10)
ax2.set_xticks(np.arange(0, 21, 2))
ax2.set_xticklabels(np.arange(0, 21, 2))
ax2.grid(linestyle=':', axis='y')
ax3.grid(linestyle=':', axis='y')
# ax2.bar(groups_1.index.to_list(), (groups_1['id'] / sum(groups_1['id']) * 100).to_list(), bar_width, color = color[0][0], align='center')
# ax3.bar(groups.index.to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), bar_width, color = color[0][0], align='center', label = "Similarity 1")
# ax2.axvline(x = mean_length, ymin = 0, ymax = 1, ls = '--', lw = 2, color = color[0][1])
# ax2.annotate(text = "Avg: {:.3f}".format(mean_length), xy = (mean_length, 24), \
#     xytext = (2.5, 21), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))

# mean_length = np.mean(length_2)
# length_2 = list(filter(lambda x: x <= 13, length_2))
# data = pd.DataFrame({"id":range(len(length_2)),"len":length_2})
# groups = data.groupby(["len"]).count()

ax2.bar((groups_2.index - bar_width * 0.5).to_list(), (groups_2['id'] / sum(groups_2['id']) * 100).to_list(), bar_width, color=color[1][0], align='center')
ax3.bar((groups_2.index - bar_width * 0.5).to_list(), (groups_2['id'] / sum(groups_2['id']) * 100).to_list(), bar_width, color=color[1][0], align='center', label = "Similarity 2")
ax2.bar((groups_1.index + bar_width * 0.5).to_list(), (groups_1['id'] / sum(groups_1['id']) * 100).to_list(), bar_width, color=color[0][0], align='center')
ax3.bar((groups_1.index + bar_width * 0.5).to_list(), (groups_1['id'] / sum(groups_1['id']) * 100).to_list(), bar_width, color=color[0][0], align='center', label = "Similarity 1")

ax2.axvline(x = mean_length_2, ymin = 0, ymax = 1., ls = '--', lw = 2, color = color[1][1])
ax3.axvline(x = mean_length_2, ymin = 0, ymax = 0.85, ls = '--', lw = 2, color = color[1][1])
ax3.annotate(text = "Avg: {:.1f}".format(mean_length_2), xy = (mean_length_2, 65), \
    xytext = (1.5, 55), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))
ax2.axvline(x = mean_length_1, ymin = 0, ymax = 0.95, ls = '--', lw = 2, color = color[0][1])
ax2.annotate(text = "Avg: {:.1f}".format(mean_length_1), xy = (mean_length_1, 22), \
    xytext = (2, 17.5), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))



ax2.set_ylim(0, 24)
ax3.set_ylim(45, 78)

ax2.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)

d = 1. 
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
              linestyle='none', color='black', mec='black', mew=1, clip_on=False)
ax2.plot([0, 1], [1, 1],transform=ax2.transAxes, **kwargs)
ax3.plot([1, 0], [0, 0], transform=ax3.transAxes, **kwargs)

ax3.legend(loc = 'upper right')
plt.savefig("../../../figures/relational_annotation_distribution.pdf", format = "pdf")
plt.clf()
