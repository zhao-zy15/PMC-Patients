from multiprocessing import Pool
import json
import os
import xml.etree.cElementTree as ET
import pandas as pd
from tqdm import trange

"""
    Extract referece from PMC OA xml.
    Input:
        file_path, PMID
    Output:
        PMID, cites (PMIDs of articles that in this article's reference)
"""
def extract_cites(msg):
    file_path, PMID = msg
    # Some xml files are corrupted.
    try:
        tree = ET.parse(os.path.join(data_dir, file_path))
        root = tree.getroot()
    except Exception as e:
        return PMID, []
    cites = []
    for ref in root.iterfind(".//ref"):
        for id_node in ref.iterfind(".//pub-id"):
            # Only consider PMID to track citation.
            if id_node is not None and 'pub-id-type' in id_node.attrib and id_node.attrib['pub-id-type'] == "pmid":
                cites.append(id_node.text)
    return PMID, cites
        
# PMC OA directory
data_dir = "../../../../PMC_OA/"
# Results from PMC_OA_meta.py
file_list = pd.read_csv(os.path.join(data_dir, "PMC_OA_meta.csv"))

msgs = [(file_list['file_path'].iloc[i], str(file_list['PMID'].iloc[i])) for i in range(len(file_list))]
pool = Pool(processes = 15)
results = pool.map(extract_cites, msgs)

PMC_cites = {}
for result in results:
    if len(result[1]) > 0:
        PMC_cites[result[0]] = result[1]

json.dump(PMC_cites, open(os.path.join(data_dir, "PMC_cites.json"), "w"), indent = 4)
