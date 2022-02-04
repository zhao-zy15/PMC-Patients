from multiprocessing import Pool
import xml.etree.cElementTree as ET
import pandas as pd
import json
import os
from tqdm import tqdm

"""
    Extract citations from PubMed xml.
    Only extract references identified by PMID of an article.
"""

def extract_ref(file_name):
    tree = ET.parse(os.path.join(data_dir, file_name))
    root = tree.getroot()
    cites = {}

    for article in root.iterfind('./PubmedArticle'):
        PMID = article.find(".//PMID").text
        cited_articles = []
        for ref in article.iterfind(".//Reference"):
            # For accuracy and simplicity, we only use PMID to track citations.
            id_node = ref.find(".//ArticleId[@IdType='pubmed']")
            if id_node != None:
                cited_articles.append(id_node.text)
        if cited_articles:
            cites[PMID] = cited_articles
    
    json.dump(cites, open(os.path.join(result_dir, file_name.replace(".xml", ".json")), "w"), indent = 4)
    del cites


if __name__ == "__main__":
    '''
    Directory to pubmed_abstract_xml downloaded from
        https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ and
        https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/
    '''
    data_dir = "../../../../pubmed/pubmed_abstract_xml"
    result_dir = "../../../../pubmed/pubmed_citation"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    file_list = os.listdir(data_dir)    
    
    # TODO: Multi-thread might be better.
    pool = Pool(processes = 30)
    pool.map(extract_ref, file_list)
