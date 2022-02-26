import json
import os
import xml.etree.cElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


patients = json.load(open("../../../meta_data/PMC-Patients.json", "r"))
pmc_path = "../../../../PMC_OA/"

type_count_art = {"None": 0}
type_count = {"None": 0}
PMIDs = set()
for patient in tqdm(patients):
    # Get article type from PMC OA.
    PMID = patient['PMID']
    file_path = patient['file_path']
    file_name = os.path.join(pmc_path, file_path)
    tree = ET.parse(file_name)
    root = tree.getroot()
    article_type = root.attrib['article-type']
    if article_type is not None:
        if article_type in type_count:
            type_count[article_type] += 1
        else:
            type_count[article_type] = 1
    else:
        type_count["None"] += 1
    # Count different types of articles at two levels: Article and Note.
    if PMID not in PMIDs:
        if article_type is not None:
            if article_type in type_count_art:
                type_count_art[article_type] += 1
            else:
                type_count_art[article_type] = 1
        else:
            type_count_art["None"] += 1
        PMIDs.add(PMID)

import ipdb; ipdb.set_trace()

# In case bug, cache the counts.
#json.dump(type_count, open("type_count.json", "w"), indent = 4)
#json.dump(type_count_art, open("type_count_art.json", "w"), indent = 4)


#type_count = json.load(open("type_count.json", "r"))
#type_count_art = json.load(open("type_count_art.json", "r"))

# Set several parameters for pyplot.
plt.rcParams['figure.figsize'] = (3.7,2.8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

# Only consider top types and take other types as 'others'.
new_dict = {"others": 0}
for k, v in type_count_art.items():
    if k == "review-article":
        new_dict[k] = v
        continue
    if v / sum(type_count_art.values()) < 0.01 or k == "other":
        new_dict['others'] += v
    else:
        new_dict[k] = v

# Calculate cumulative distribution.
labels = ['case-report', 'research-article', 'review-article', 'others']
cum_y_art = [new_dict[labels[0]] / sum(new_dict.values())]
for label in labels[1:]:
    cum_y_art.append(new_dict[label] / sum(new_dict.values()) + cum_y_art[-1])

new_dict = {"others": 0}
for k, v in type_count.items():
    if v / sum(type_count.values()) < 0.01 or k == "other":
        new_dict['others'] += v
    else:
        new_dict[k] = v

cum_y = [new_dict[labels[0]] / sum(new_dict.values())]
for label in labels[1:]:
    cum_y.append(new_dict[label] / sum(new_dict.values()) + cum_y[-1])


# Plot stack bar plot.
width = 0.15
colors = ['steelblue', 'indianred', 'gold', 'lightgreen']
plt.xlim(0, 1)
for i in [3,2,1,0]:
    plt.bar([0.16, 0.41], [cum_y_art[i], cum_y[i]], width = width, color = colors[i], label = labels[i])
plt.xticks([0.16, 0.41], ["Article", "Note"])
plt.legend(fontsize = 10)
plt.tight_layout(pad = 0.01)
plt.savefig("../../figures/article_type_distribution.pdf", format = "pdf")
