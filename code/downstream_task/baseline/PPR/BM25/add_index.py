from elasticsearch import Elasticsearch as ES
import json
import os
from tqdm import tqdm


corpus_path = "../../../../../datasets/PPR/corpus.jsonl"
corpus = []
with open(corpus_path, 'r') as f:
    for line in f:
        corpus.append(json.loads(line))

es = ES()
if not es.indices.exists(index = "ppr_corpus"):
    es.indices.create(index = "ppr_corpus")
train_body = []
for patient in tqdm(corpus):
    train_body.append({"index": {"_index": "ppr_corpus", "_id": patient["_id"]}})
    train_body.append({"text": patient['text']})

    if len(train_body) >= 10000:
        es.bulk(operations = train_body)
        train_body = []
es.bulk(operations = train_body)

print(es.count(index = 'ppr_corpus'))

#import ipdb; ipdb.set_trace()
