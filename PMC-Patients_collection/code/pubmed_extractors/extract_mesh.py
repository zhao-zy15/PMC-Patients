from multiprocessing import Pool
import xml.etree.cElementTree as ET
import json
import os
from tqdm import tqdm

"""
    Extract Mesh terms from PubMed xml.
    Note only a small amount of articles are provided with Mesh terms.
"""

def extract_mesh(file_name):
    global PMIDs
    tree = ET.parse(os.path.join(data_dir, file_name))
    root = tree.getroot()
    PMID2Mesh = {}

    for article in root.iterfind('./PubmedArticle'):
        PMID = article.find(".//PMID").text
        if PMID in PMIDs:
            Meshes = []
            for mesh in article.iterfind(".//MeshHeading"):
                Meshes.append(mesh[0].text)
            if Meshes:
                PMID2Mesh[PMID] = Meshes
    
    return PMID2Mesh


if __name__ == "__main__":
    # PMIDS to extract Mesh terms.
    PMIDs = json.load(open("../../meta_data/PMIDs.json", "r"))
    PMIDs = set(PMIDs)
    print(len(PMIDs))
    data_dir = "../../../../pubmed/pubmed_abstract_xml"
    file_list = os.listdir(data_dir)    

    PMID2Mesh = {}
    
    pool = Pool(processes = 20)
    results = pool.map(extract_mesh, file_list)
    for result in results:
        PMID2Mesh.update(result)

    #import ipdb; ipdb.set_trace()
    json.dump(PMID2Mesh, open("../../meta_data/PMID2Mesh.json", "w"), indent = 4)
