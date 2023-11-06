import json
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval


BM25 = json.load(open("../PAR_BM25_test.json", "r"))
BM25 = {query: {k: i+1 for i, (k,v) in enumerate(sorted(BM25[query].items(), key=lambda x: x[1], reverse=True))} for query in BM25}
dense = json.load(open("../PAR_pubmed_test.json", "r"))
dense = {query: {k: i+1 for i, (k,v) in enumerate(sorted(dense[query].items(), key=lambda x: x[1], reverse=True))} for query in dense}
dense2 = json.load(open("../PAR_link_test.json", "r"))
dense2 = {query: {k: i+1 for i, (k,v) in enumerate(sorted(dense2[query].items(), key=lambda x: x[1], reverse=True))} for query in dense2}

for rrf_k in [0, 2, 5, 10, 20, 50, 60, 80, 100, 200, 500]:
    results = {}
    for query in tqdm(BM25):
        results[query] = {k: 1 / (rrf_k + v) for k,v in BM25[query].items()}
        for doc in dense[query]:
            if doc in results[query]:
                results[query][doc] += 1 / (rrf_k + dense[query][doc])
            else:
                results[query][doc] = 1 / (rrf_k + dense[query][doc])
        for doc in dense2[query]:
            if doc in results[query]:
                results[query][doc] += 1 / (rrf_k + dense2[query][doc])
            else:
                results[query][doc] = 1 / (rrf_k + dense2[query][doc])

    qrels = {}
    with open("../../../../../datasets/PAR/qrels_test.tsv", "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        q, doc, score = line.split('\t')
        if q in qrels:
            qrels[q][doc] = int(score)
        else:
            qrels[q] = {doc: int(score)}

    evaluation = EvaluateRetrieval()
    metrics = evaluation.evaluate(qrels, results, [10, 1000])
    mrr = evaluation.evaluate_custom(qrels, results, [10000], metric="mrr")
    print(rrf_k)
    print(mrr[f'MRR@{10000}'], metrics[3]['P@10'], metrics[0]['NDCG@10'], metrics[2]['Recall@1000'])
