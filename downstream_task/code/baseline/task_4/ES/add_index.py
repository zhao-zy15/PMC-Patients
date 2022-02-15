from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm


es = ES()

data_dir = "../../../../../../pubmed/pubmed_title_abstract/"
PMIDs = set()
for file_name in tqdm(os.listdir(data_dir)):
    body = []
    articles = json.load(open(os.path.join(data_dir, file_name), "r"))
    for PMID in articles:
        if PMID not in PMIDs:
            body.append({"index": {"_index": "pubmed_title_abstract", "_id": PMID}})
            body.append({"title": articles[PMID]['title'], "abstract": articles[PMID]['abstract']})
            PMIDs.add(PMID)
    es.bulk(body)

print(es.count(index = 'pubmed_title_abstract'))

import ipdb; ipdb.set_trace()