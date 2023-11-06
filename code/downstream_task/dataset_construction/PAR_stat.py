import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

PAR_train = pd.read_csv("../../../datasets/PAR/qrels_train.tsv", sep = '\t')
scores = PAR_train['score']
queries = pd.unique(PAR_train['query-id'])
print("Labels count: ", len(scores), sum(scores == 2), sum(scores == 1))
print("Average labels per note", len(scores) / len(queries), sum(scores == 2) / len(queries), sum(scores == 1) / len(queries))
print("Queries: ", len(queries))

PAR_dev = pd.read_csv("../../../datasets/PAR/qrels_dev.tsv", sep = '\t')
scores = PAR_dev['score']
queries = pd.unique(PAR_dev['query-id'])
print("Labels count: ", len(scores), sum(scores == 2), sum(scores == 1))
print("Average labels per note", len(scores) / len(queries), sum(scores == 2) / len(queries), sum(scores == 1) / len(queries))
print("Queries: ", len(queries))


PAR_test = pd.read_csv("../../../datasets/PAR/qrels_test.tsv", sep = '\t')
scores = PAR_test['score']
queries = pd.unique(PAR_test['query-id'])
print("Labels count: ", len(scores), sum(scores == 2), sum(scores == 1))
print("Average labels per note", len(scores) / len(queries), sum(scores == 2) / len(queries), sum(scores == 1) / len(queries))
print("Queries: ", len(queries))
