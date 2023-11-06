
## Downstream Tasks
Code for collection downstream tasks dataset and several baselines.

### Patient-Patient Retrieval (PPR)
Split by `dataset_split.py`. PPR dataset is a dict where the keys are `patient_uid` of queries and each entry is a list of `patient_uid`, representing similar patients to the query.

### Patient-Article Retrieval (PAR)
Split by `dataset_split.py`. PPR dataset is a dict where the keys are `patient_uid` of queries and each entry is a list of `PMID`, representing articles relevant to the query.


## Baseline Models
Codes for build and evaluate baseline models for downstream tasks are in directory `baseline`. Evaluation codes are reusable.
To simply reproduce baselines, download our dataset and unzip it to directory `datasets/`.

### PPR

**BM25**

Elasticsearch and the python package Elasticsearch is required.
Add index.
```
python baseline/PAR/BM25/add_index.py
```
Query with multithreads.
```
python baseline/PAR/BM25/query.py
```

**Dense Retrieval**

For zero-shot retrieval,
```
python baseline/PPR/Dense/sbert.py
```
For fine-tuning models,
```
python baseline/PPR/Dense/main.py
```
For evaluating fine-tuned models,
```
python baseline/PPR/Dense/generate_embeddings.py
```

**Nearest Neighbor (NN)**
```
python baseline/PPR/NN/NN.py
```

**RRF**
```
python baseline/PPR/RRF/RRF.py
```
