import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


meta_data_dir = "../../meta_data"

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

plt.rcParams['figure.figsize'] = (3.7,2.8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.xlabel('Number of relevance annotation per note', fontsize=12)
plt.ylabel('Percent of notes', fontsize=12)
plt.tick_params(labelsize=10)
plt.xticks(ticks = np.arange(0, 101, 10), labels = np.arange(0, 101, 10))
plt.grid(linestyle=':', axis='y')
plt.bar(groups.index.to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), color='steelblue', align='center')
plt.axvline(x = mean_length, ymin = 0, ymax = 0.77, ls = '--', lw = 1, color = 'darkred')
plt.annotate(text = "Avg. {:.2f}".format(mean_length), xy = (18.6, 5.2), \
    xytext = (23, 5.6), fontsize = 12, arrowprops = dict(arrowstyle = '->', color = 'black'))
plt.tight_layout(pad = 0.01)
plt.savefig("../../figures/patient-article_relevance_distribution.pdf", format = "pdf")
plt.clf()


patient_similarity = json.load(open(os.path.join(meta_data_dir, "patient2patient_similarity.json"), "r"))

print("====== Patient Similarity=======")
length = [len(item[0]) for item in patient_similarity.values()]
print("Similarity 1 count: ", sum(length))
length = [len(item[1]) for item in patient_similarity.values()]
print("Similarity 2 count: ", sum(length))
length = [len(item[0]) + len(item[1]) for item in patient_similarity.values()]
print("Similarity total count: ", sum(length))
print("Average similarity per note: ", np.mean(length))
print("Max Relevance per article: ", np.max(length))

print("Length greater than 20: ", len(list(filter(lambda x: x > 20, length))))
mean_length = np.mean(length)
length = list(filter(lambda x: x <= 20, length))
data = pd.DataFrame({"id":range(len(length)),"len":length})
groups = data.groupby(["len"]).count()

plt.rcParams['figure.figsize'] = (3.7,2.8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.xlabel('Number of similarity annotation per note', fontsize=12)
plt.ylabel('Percent of notes', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':', axis='y')
plt.bar(groups.index.to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), color='steelblue', align='center')
plt.axvline(x = mean_length, ymin = 0, ymax = 0.65, ls = '--', lw = 1, color = 'darkred')
plt.annotate(text = "Avg. {:.2f}".format(mean_length), xy = (1.7, 25), \
    xytext = (3, 31), fontsize = 12, arrowprops = dict(arrowstyle = '->', color = 'black'))
plt.tight_layout(pad = 0.01)
plt.savefig("../../figures/patient-patient_similarity_distribution.pdf", format = "pdf")
plt.clf()
