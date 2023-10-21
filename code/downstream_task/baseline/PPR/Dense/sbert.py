from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader


corpus_path = "../../../../../datasets/PPR/corpus.jsonl"
query_path = "../../../../../datasets/queries/test_queries.jsonl"
qrels_path = "../../../../../datasets/PPR/qrels_test.tsv"

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path,
    query_file=query_path,
    qrels_file=qrels_path).load_custom()


model = DRES(models.SentenceBERT("./contriever-msmarco"), batch_size=128)
retriever = EvaluateRetrieval(model, score_function="dot")
results = retriever.retrieve(corpus, queries)
metrics = retriever.evaluate(qrels, results, [10, 1000])
mrr = retriever.evaluate_custom(qrels, results, [len(corpus)], metric="mrr")
print(metrics)
print(mrr)
