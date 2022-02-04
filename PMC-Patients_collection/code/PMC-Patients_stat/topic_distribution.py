import json
from tqdm import tqdm
import xml.etree.cElementTree as ET
import os
from xml_utils import getText
import re


PMID2keyword = json.load(open("../../meta_data/PMID2keywords.json", "r"))
PMID2Mesh = json.load(open("../../meta_data/PMID2Mesh.json", "r"))
Mesh2tree = json.load(open("../../../PMC_OA/Mesh2tree.json", "r"))
Mesh_count = {}
for PMID in PMID2Mesh.keys():
    for Mesh in PMID2Mesh[PMID]:
        if Mesh in Mesh2tree:
            if Mesh in Mesh_count:
                Mesh_count[Mesh] += 1
            else:
                Mesh_count[Mesh] = 1

Mesh_distribution = {Mesh: Mesh_count[Mesh] / len(PMID2Mesh) for Mesh in Mesh_count}
Mesh_distribution = sorted(Mesh_distribution.items(), key = lambda x: x[1], reverse=True)
json.dump(Mesh_distribution, open("../../meta_data/Mesh_distribution.json", "w"), indent = 4)

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
import ipdb; ipdb.set_trace()