import numpy as np
import json
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


embeddings = np.load("embeddings.npy")
embeddings = embeddings / np.linalg.norm(embeddings, axis = 1).reshape(-1, 1)
np.random.seed(21)
# Randomly sample some embeddings and calculate pairwise distances.
num = 10000
sample_index = np.random.randint(0, embeddings.shape[0], size = num)
embeddings = embeddings[sample_index, :]

distances = {}
for i in trange(num):
    for j in range(i + 1, num):
        # Only reserve 3 digits and cache it as integer.
        dis = int(np.dot(embeddings[i], embeddings[j]) * 1000)
        if dis == "-0.000":
            dis = "0.000"
        if dis in distances:
            distances[dis] += 1
        else:
            distances[dis] = 1

#json.dump(distances, open("embedding_distances.json", "w"), indent = 4)
#print(sum(distances.values()))


#distances = json.load(open("embedding_distances.json", "r"))
total = sum(distances.values())

int_dist = []
for dis in distances:
    tick = int(float(dis) * 1000)
    int_dist += [tick] * distances[dis]

mean = np.mean(int_dist)
# Truncate the distribution and count.
int_dist = list(filter(lambda x: x > 600, int_dist))
data = pd.DataFrame({"id":range(len(int_dist)),"len":int_dist})
groups = data.groupby(["len"]).count()

plt.rcParams['figure.figsize'] = (3.7, 3)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.bar((groups.index).to_list(), (groups['id'] / sum(groups['id']) * 100).to_list(), color = "steelblue", align = "center")
plt.xticks(ticks = [600, 700, 800, 900, 1000], labels = ["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize = 10)
plt.xlabel("Cosine similarity", fontsize = 12)
plt.grid(linestyle=':', axis='y')
plt.axvline(x = mean, ymin = 0, ymax = 0.4, ls = '--', lw = 1.5, color = "turquoise")
plt.annotate(text = "Avg. {:.2f}".format(mean / 1000), xy = (mean, 1.1), \
    xytext = (830, 1.05), fontsize = 10, arrowprops = dict(arrowstyle = '->', color = 'black'))
plt.text(550, 2.8, "%", fontsize = 10)

plt.tight_layout(pad = 0.01)
plt.savefig("embedding_distance_distribution.pdf", format = "pdf")

