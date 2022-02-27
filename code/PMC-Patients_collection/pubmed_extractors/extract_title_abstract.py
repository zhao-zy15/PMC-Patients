from multiprocessing import Pool
import xml.etree.cElementTree as ET
import pandas as pd
import json
import os
from tqdm import tqdm
import sys
sys.path.append("..")
from xml_utils import getText

"""
    Extract article title and abstract from PubMed xml.
    Articles without *both* title and abstact are filtered.
"""

def extract_title_abstract(file_name):
    tree = ET.parse(os.path.join(data_dir, file_name))
    root = tree.getroot()
    content = {}

    for article in root.iterfind('./PubmedArticle'):
        PMID = article.find(".//PMID").text
        abstract_node = article.find(".//Abstract")
        # Note that some articles in PubMed miss abstract and title, and are thus filtered.
        if abstract_node is not None:
            abstract = getText(abstract_node)
        else:
            abstract = ""
        title_node = article.find(".//ArticleTitle")
        if title_node is not None:
            title = getText(title_node)
        else:
            title = ""
        if len(abstract) + len(title) > 0:
            content[PMID] = {"title": title, "abstract": abstract}
        
    json.dump(content, open(os.path.join(result_dir, file_name.replace(".xml", ".json")), "w"), indent = 4)
    return len(content)


if __name__ == "__main__":
    '''
    Directory to pubmed_abstract_xml downloaded from
        https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ and
        https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/
    '''
    data_dir = "../../../../pubmed/pubmed_abstract_xml"
    result_dir = "../../../../pubmed/pubmed_title_abstract"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    file_list = os.listdir(data_dir)    
    
    # TODO: Multi-thread might be better.
    pool = Pool(processes = 30)
    results = pool.map(extract_title_abstract, file_list)

    abstract_count = 0
    for result in results:
        abstract_count += result
    # Number of total articles extracted.
    print(abstract_count)
    
