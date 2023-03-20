from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm


es = ES()
es.indices.create(index = "par_abstract")

PAR_abstracts = json.load(open("../../../../../datasets/patient2article_retrieval/PAR_abstracts.json", "r"))
body = []
for PMID in tqdm(PAR_abstracts):
    body.append({"index": {"_index": "par_abstract", "_id": PMID}})
    body.append({"title": PAR_abstracts[PMID]['title'], "abstract": PAR_abstracts[PMID]['abstract']})
    if len(body) >= 10000:
        es.bulk(body)
        body = []
es.bulk(body)

print(es.count(index = 'par_abstract'))
