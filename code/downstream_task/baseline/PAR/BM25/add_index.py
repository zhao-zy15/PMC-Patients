from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm


es = ES("https://localhost:9200", 
        ca_certs = "/media/sdb/ZhengyunZhao/elasticsearch-8.8.2/config/certs/http_ca.crt",
        basic_auth = ("elastic", "Opi-UQJDyfSXjDSTmHnB"))

if not es.indices.exists(index = "par_corpus"):
    es.indices.create(index = "par_corpus")

corpus_path = "../../../../../datasets/PAR/corpus.jsonl"
corpus = []
with open(corpus_path, 'r') as f:
    for line in f:
        corpus.append(json.loads(line))

body = []
for article in tqdm(corpus):
    body.append({"index": {"_index": "par_corpus", "_id": article['_id']}})
    body.append({"title": article['title'], "text": article['text']})
    if len(body) >= 100000:
        es.bulk(operations = body)
        body = []
es.bulk(operations = body)

print(es.count(index = 'par_corpus'))
