import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

PPR_train = pd.read_csv("../../../datasets/PPR/qrels_train.tsv", sep = '\t')
scores = PPR_train['score']
queries = pd.unique(PPR_train['query-id'])
print("Labels count: ", len(scores), sum(scores == 2), sum(scores == 1))
print("Average labels per note", len(scores) / len(queries), sum(scores == 2) / len(queries), sum(scores == 1) / len(queries))
print("Queries: ", len(queries))

PPR_dev = pd.read_csv("../../../datasets/PPR/qrels_dev.tsv", sep = '\t')
scores = PPR_dev['score']
queries = pd.unique(PPR_dev['query-id'])
print("Labels count: ", len(scores), sum(scores == 2), sum(scores == 1))
print("Average labels per note", len(scores) / len(queries), sum(scores == 2) / len(queries), sum(scores == 1) / len(queries))
print("Queries: ", len(queries))


PPR_test = pd.read_csv("../../../datasets/PPR/qrels_test.tsv", sep = '\t')
scores = PPR_test['score']
queries = pd.unique(PPR_test['query-id'])
print("Labels count: ", len(scores), sum(scores == 2), sum(scores == 1))
print("Average labels per note", len(scores) / len(queries), sum(scores == 2) / len(queries), sum(scores == 1) / len(queries))
print("Queries: ", len(queries))
