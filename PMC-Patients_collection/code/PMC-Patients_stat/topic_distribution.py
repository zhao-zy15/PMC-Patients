import json
from tqdm import tqdm
import xml.etree.cElementTree as ET
import os
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


def plot_distribution(frequency, file_name):
    frequency.reverse()
    terms = [freq[0] for freq in frequency]
    values = [freq[1] for freq in frequency]
    plt.rcParams['figure.figsize'] = (5.0, 6.3)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.xlabel('Frequency of the MeSH terms', fontsize=14)
    plt.tick_params(labelsize=10)

    norm = plt.Normalize(-3, 1.1 * len(values))
    norm_values = norm(range(len(values)))
    map_vir = cm.get_cmap(name = 'rainbow')
    colors = map_vir(norm_values)

    plt.grid(linestyle=':', axis='x')
    plt.barh(range(len(frequency)), values, color=colors, align='center')
    plt.yticks(range(len(frequency)), terms, fontsize = 12)

    plt.tight_layout(pad = 0.5)
    plt.savefig(file_name, format = "pdf")
    plt.clf()


patients = json.load(open("../../meta_data/PMC-Patients.json", "r"))
notes_in_PMID = {}
for patient in patients:
    if patient['PMID'] in notes_in_PMID:
        notes_in_PMID[patient['PMID']] += 1
    else:
        notes_in_PMID[patient['PMID']] = 1

figure_path = "../../figures"
PMID2Mesh = json.load(open("../../meta_data/PMID2Mesh.json", "r"))
Mesh2tree = json.load(open("../../../../PMC_OA/Mesh2tree.json", "r"))
Mesh_count = {}
total_notes = 0
for PMID in tqdm(PMID2Mesh):
    for Mesh in PMID2Mesh[PMID]:
        if Mesh in Mesh2tree:
            if Mesh in Mesh_count:
                Mesh_count[Mesh] += notes_in_PMID[PMID]
            else:
                Mesh_count[Mesh] = notes_in_PMID[PMID]
    total_notes += notes_in_PMID[PMID]

Mesh_distribution = {Mesh: Mesh_count[Mesh] / total_notes for Mesh in Mesh_count}
Mesh_distribution = sorted(Mesh_distribution.items(), key = lambda x: x[1], reverse=True)
plot_distribution(Mesh_distribution[:30], os.path.join(figure_path, "Mesh_distribution_top30.pdf"))
import ipdb; ipdb.set_trace()



'''
def wordcloud_to_svg(frequency_dict, file_name):
    wc = WordCloud(width = 900, height = 600, background_color = "white", min_font_size = 22, \
        max_font_size = 100, colormap = "Dark2", font_path = "../../../../Gill_Sans_MT.ttf", random_state = 21)
    wc.fit_words(frequency_dict)
    svg = wc.to_svg(embed_font = True)
    with open(os.path.join(figure_path, file_name + ".svg"), "w") as f:
        f.write(svg)
    wc.to_file(os.path.join(figure_path, file_name + ".pdf"))


figure_path = "../../figures"
sw = stopwords.words("english")
stemmer = PorterStemmer()
non_word = re.compile(r'[^a-z ]')


PMID2Mesh = json.load(open("../../meta_data/PMID2Mesh.json", "r"))
Mesh_count = {}
for PMID in tqdm(PMID2Mesh):
    for Mesh in PMID2Mesh[PMID]:
        if Mesh in Mesh_count:
            Mesh_count[Mesh] += 1
        else:
            Mesh_count[Mesh] = 1

Mesh_distribution = {Mesh: Mesh_count[Mesh] / len(PMID2Mesh) for Mesh in Mesh_count}
Mesh_distribution = sorted(Mesh_distribution.items(), key = lambda x: x[1], reverse=True)
import ipdb; ipdb.set_trace()


Mesh_words_count = {}
for PMID in tqdm(PMID2Mesh):
    for Mesh in PMID2Mesh[PMID]:
        Mesh = non_word.sub('', Mesh.lower())
        Mesh_words = [stemmer.stem(word) for word in Mesh.split() if word not in sw]
        for word in Mesh_words:
            if word in Mesh_words_count:
                Mesh_words_count[word] += 1
            else:
                Mesh_words_count[word] = 1

wordcloud_to_svg(Mesh_words_count, "Mesh_wordcloud")


PMID2keyword = json.load(open("../../meta_data/PMID2keywords.json", "r"))
kwd_words_count = {}
for PMID in tqdm(PMID2keyword):
    for keyword in PMID2keyword[PMID]:
        keyword = non_word.sub('', keyword.lower())
        keyword_words = [stemmer.stem(word) for word in keyword.split() if word not in sw]
        for word in keyword_words:
            if word in kwd_words_count:
                kwd_words_count[word] += 1
            else:
                kwd_words_count[word] = 1

wordcloud_to_svg(kwd_words_count, "Keywords_wordcloud")


patients = json.load(open("../../meta_data/PMC-Patients.json", "r"))
text_words_count = {}
for patient in tqdm(patients):
    text = non_word.sub('', patient['patient'].lower())
    text_words = [stemmer.stem(word) for word in text.split() if word not in sw]
    for word in text_words:
        if word in text_words_count:
            text_words_count[word] += 1
        else:
            text_words_count[word] = 1

wordcloud_to_svg(text_words_count, "Texts_wordcloud")
'''



'''
Mesh2tree = json.load(open("../../../../PMC_OA/Mesh2tree.json", "r"))
Mesh_count = {}
for PMID in tqdm(PMID2Mesh):
    for Mesh in PMID2Mesh[PMID]:
        if Mesh in Mesh2tree:
            if Mesh in Mesh_count:
                Mesh_count[Mesh] += 1
            else:
                Mesh_count[Mesh] = 1

Mesh_distribution = {Mesh: Mesh_count[Mesh] / len(PMID2Mesh) for Mesh in Mesh_count}
Mesh_distribution = sorted(Mesh_distribution.items(), key = lambda x: x[1], reverse=True)
#json.dump(Mesh_distribution, open("../../meta_data/Mesh_distribution.json", "w"), indent = 4)
'''


'''
keyword_count = {}
label_pattern = re.compile(r'^[0-9]\.[0-9]?\.?[0-9]?\.? ')
for PMID in PMID2keyword.keys():
    for keyword in PMID2keyword[PMID]:
        temp = label_pattern.sub('', keyword)
        if temp in keyword_count:
            keyword_count[temp] += 1
        else:
            keyword_count[temp] = 1

del keyword_count['']

keyword_distribution = {keyword: keyword_count[keyword] / len(PMID2keyword) for keyword in keyword_count}
keyword_distribution = sorted(keyword_distribution.items(), key = lambda x: x[1], reverse=True)
json.dump(keyword_distribution, open("../../meta_data/keyword_distribution.json", "w"), indent = 4)
'''
