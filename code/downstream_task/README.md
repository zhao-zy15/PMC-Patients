
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
python baseline/task_3/ES/add_index.py
```
Query with multithreads.
```
python baseline/task_3/ES/query.py
```

**EBR**

First generate embeddings for each query and document.
```
python baseline/task_3/EBR/embedding.py
```
Then perform embedding based retrieval.
```
python baseline/task_3/EBR/EBR.py
```

**DSC**
```
python baseline/task_3/Dice/dice.py
```


### PAR
Note that to run baseline models for PAR, you have to download PubMed and extract titles and abstracts (run `code/PMC-Patients_collection/pubmed_extractors/extract_title_abstract.py`).

**BM25**

Elasticsearch and the python package Elasticsearch is required.
Add index.
```
python baseline/task_4/ES/add_index.py
```
Query with multithreads.
```
python baseline/task_4/ES/query.py
```

**KNN**

```
python baseline/task_4/KNN/KNN.py
```

