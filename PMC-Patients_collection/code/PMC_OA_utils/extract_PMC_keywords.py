from multiprocessing import Pool
import json
import os
import xml.etree.cElementTree as ET
from tqdm import trange
import sys
sys.path.append("..")
from xml_utils import getText

"""
    Extract referece from PMC OA xml (only articles in PMC-Patients).
    Input:
        file_path, PMID
    Output:
        PMID, keywords.
"""
def extract_keywords(msg):
    file_path, PMID = msg
    
    tree = ET.parse(os.path.join(data_dir, file_path))
    root = tree.getroot()

    keywords = []
    for kwd in root.iterfind(".//kwd"):
        keywords.append(getText(kwd))
    return PMID, keywords
        
# PMC OA directory
data_dir = "../../../../PMC_OA/"
# Results from PMC_OA_meta.py
patients = json.load(open("../../meta_data/PMC-Patients.json", "r"))

msgs = [(patient['file_path'], patient['PMID']) for patient in patients]
pool = Pool(processes = 15)
results = pool.map(extract_keywords, msgs)

PMID2keywords = {}
for result in results:
    if len(result[1]) > 0:
        PMID2keywords[result[0]] = result[1]

json.dump(PMID2keywords, open("../../meta_data/PMID2keywords.json", "w"), indent = 4)
