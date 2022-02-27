
## Downstream Tasks
Code for collection downstream tasks dataset and several baselines.

### Patient Note Recognition (PNR)
Generate whole PNR dataset (without train/dev/test split) by 
```
python dataset_construction/patient_note_recognition.py
```
Train/dev/test split is done by
```
python dataset_construction/dataset_split.py
```
PNR dataset is a list of dict with keys:
- `PMID`: string. PMID of the article.
- `file_path`: string. File path of xml in PMC OA.
- `title`, `abstract`: string. Title and abstract of the article.
- `texts`: list of string. Each entry is one paragraph of the article.
- `tags`: list of string. BIOES tags. Each entry indicates tag for corresponding paragraph.

### Patient-Patient Similarity (PPS)
**After** running `dataset_split.py`, sample negative instances for PPS.
```
python dataset_construction/PPS_graph.py
```
PPS dataset is a list of triplets. Each entry is in format (patient_uid_1, patient_uid_2, similarity) where similarity has three values:0, 1, 2, indicating corresponding similarity.

### Patient-Patient Retrieval (PPR)
Split by `dataset_split.py`. PPR dataset is a dict where the keys are `patient_uid` of queries and each entry is two lists of `patient_uid`, representing notes of similarity 1 and 2, respectively.

### Patient-Article Retrieval (PAR)
Split by `dataset_split.py`. PPR dataset is a dict where the keys are `patient_uid` of queries and each entry is a list of `PMID`, representing articles relevant to the query.


## Baseline Models
Codes for build and evaluate baseline models for downstream tasks are in directory `baseline`. Evaluation codes are reusable.
To simply reproduce baselines, download xxxxx and unzip it to directory `datasets/`.
### PNR

**Demographic_finder:**

```
python baseline/task_1/demographic_finder/demographic_finder.py
```

**BERT:**

`train` argument controls whether to train the model. For other parameters, see code.
```
python baseline/task_1/BERT/main.py --train
```
**CNN+LSTM+CRF:**

`train` argument controls whether to train the model. For other parameters, see code.
```
python baseline/task_1/CNN_LSTM_CRF/main.py --train
```

### PPS

**Feature-based:**

First use Scispacy (https://github.com/allenai/scispacy) to detect named entities in patient notes and link each entity to UMLS.
```
python baseline/task_2/feature_based/NER.py
```
Then train a model with grid search for best hyperparameters.
`model` argument controls whether to use logistic regression or SVM (`LR` or `SVM`).
```
python baseline/task_2/feature_based/model.py --model LR
```

**BERT**

`train` argument controls whether to train the model. For other parameters, see code.
```
python baseline/task_2/BERT/main.py --train
```

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

