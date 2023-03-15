import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


meta_data_dir = "../../../meta_data"

relevant_article = json.load(open(os.path.join(meta_data_dir, "patient2article_relevance.json"), "r"))

print("====== Article Relevance =======")
print("Article count: ", len(relevant_article))
length = [len(item) for item in relevant_article.values()]
print("Relevance count: ", sum(length))
print("Average Relevance per article: ", np.mean(length))
print("Min Relevance per article: ", np.min(length))
print("Number of 1:", np.bincount(length)[1])
print("Max Relevance per article: ", np.max(length))

print("Length greater than 100: ", len(list(filter(lambda x: x > 100, length))))
mean_length = np.mean(length)
length = list(filter(lambda x: x <= 100, length))
data = pd.DataFrame({"id":range(len(length)),"len":length})
groups = data.groupby(["len"]).count()

plt.rcParams['figure.figsize'] = (8.5,2)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

ax1 = plt.axes([.05, 0.215, .4, .78])
ax2 = plt.axes([.56, 0.215, .4, .523])
ax3 = plt.axes([.56, 0.784, .4, .211])

ax1.set_xlabel('Number of relevance annotations', fontsize=12)
ax1.set_ylabel('Percent of patients', fontsize=12)
ax1.tick_params(labelsize=10)
ax1.set_xticks(np.arange(0, 101, 10))
ax1.set_xticklabels(np.arange(0, 101, 10))
ax1.grid(linestyle=':', axis='y')
ax1.bar(groups.index.to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), color='steelblue', align='center')
ax1.axvline(x = mean_length, ymin = 0, ymax = 0.76, ls = '--', lw = 1.5, color = 'turquoise')
ax1.annotate(text = "Avg. {:.2f}".format(mean_length), xy = (18.6, 5.1), \
    xytext = (23, 5.6), fontsize = 12, arrowprops = dict(arrowstyle = '->', color = 'black'))


patient_similarity = json.load(open(os.path.join(meta_data_dir, "patient2patient_similarity.json"), "r"))

print("====== Patient Similarity=======")
length = [len(item) for item in patient_similarity.values()]
print("Similarity count: ", sum(length))
print("Average similarity per note: ", np.mean(length))
print("Max Relevance per article: ", np.max(length))

#print("Length greater than 20: ", len(list(filter(lambda x: x > 20, length))))
mean_length = np.mean(length_1)
length_1 = list(filter(lambda x: x <= 13, length_1))
data = pd.DataFrame({"id":range(len(length_1)),"len":length_1})
groups = data.groupby(["len"]).count()

color = [["steelblue", "turquoise"], ["firebrick", "pink"]]
bar_width = 0.5
ax2.set_xlabel('Number of similarity annotations', fontsize=12)
ax2.set_ylabel('Percent of patients', fontsize=12, y = 0.746)
ax2.tick_params(labelsize=10)
ax3.tick_params(labelbottom = False, bottom = False, labelsize = 10)
ax2.set_xticks(np.arange(0, 21, 2))
ax2.set_xticklabels(np.arange(0, 21, 2))
ax2.grid(linestyle=':', axis='y')
ax3.grid(linestyle=':', axis='y')
ax2.bar(groups.index.to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), bar_width, color = color[0][0], align='center')
ax3.bar(groups.index.to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), bar_width, color = color[0][0], align='center', label = "Similarity 1")
ax2.axvline(x = mean_length, ymin = 0, ymax = 1, ls = '--', lw = 2, color = color[0][1])
ax2.annotate(text = "Avg. {:.3f}".format(mean_length), xy = (mean_length, 24), \
    xytext = (2.5, 21), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))

mean_length = np.mean(length_2)
length_2 = list(filter(lambda x: x <= 13, length_2))
data = pd.DataFrame({"id":range(len(length_2)),"len":length_2})
groups = data.groupby(["len"]).count()

ax2.bar((groups.index + bar_width).to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), bar_width, color=color[1][0], align='center')
ax3.bar((groups.index + bar_width).to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), bar_width, color=color[1][0], align='center', label = 'Similarity 2')
ax2.axvline(x = mean_length, ymin = 0, ymax = 1., ls = '--', lw = 2, color = color[1][1])
ax3.axvline(x = mean_length, ymin = 0, ymax = 0.9, ls = '--', lw = 2, color = color[1][1])
ax3.annotate(text = "Avg. {:.3f}".format(mean_length), xy = (mean_length, 70), \
    xytext = (2, 63), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))


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
plt.savefig("../../../figures/relational_annotation_distribution.pdf")
plt.clf()
