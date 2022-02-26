# PMC-Patients
This repository contains PMC-Patients dataset (including patient notes, patient-patient similarity annotations, patient-article relevance annotations and four downstream task datasets: patient note recognition PNR, patient-patient similarity PPS, patient-patient retrieval PPR, patient-article retrieval PAR), codes for collection datasets and several baseline models.

## PMC OA and PubMed Downloads
For those who only wish to reproduce baseline models, only PubMed abstracts are required for PAR task.
If you have already downloaded PMC OA and PubMed abstracts on your device, skip this step and change relative directory in later steps. Otherwise, download PMC OA and PubMed via https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/ and https://ftp.ncbi.nlm.nih.gov/pubmed/, respectively. Note that file `PMC-ids.csv` under directory https://ftp.ncbi.nlm.nih.gov/pub/pmc/ is also required.


## PMC-Patients Collection
Codes for PMC-Patients collection are in directory `PMC-Patients_collection/code/` and dataset is stored in `meta_data/`.
### Preprocessing
Enumerate all .csv files in PMC OA, filtering those without PMID, and write file_path, PMID, License of each article in a new file.
```
python PMC-Patients_collection/PMC_OA_utils/PMC_OA_meta.py
```
Extract citations in PMC OA.
```
python PMC-Patients_collection/PMC_OA_utils/extract_PMC_cites.py
```
Extract citations in PubMed.
```
python PMC-Patients_collection/pubmed_extractors/extract_cites.py
```
Extract titles and abstracts in PubMed for PAR. (If you only wish to reproduce baseline models, run this command is enough.)
```
python PMC-Patients_collection/pubmed_extractors/extract_title_abstract.py
```

### Collection
Detect and extract patient notes in PMC OA.
```
python PMC-Patients_collection/code/PMC-Patients_collection/extractor.py
```
Filter non-patient-note.
```
python PMC-Patients_collection/code/PMC-Patients_collection/filters.py
```
Annotate similarity and relevance.
```
python PMC-Patients_collection/code/PMC-Patients_collection/annotate.py
```
The results are stored in `meta_data/`. 

Patients notes are stored in `PMC-Patients.json`, which is a list of dict with keys:
- `patient_id`: string. A continuous id of patients, starting from 0.
- `patient_uid`: string. Unique ID for each patient, with format PMID-x, where PMID is the PubMed Identifier of source article of the note and x denotes index of the note in source article.
- `PMID`: string. PMID for source article.
- `file_path`: string. File path of xml file of source article.
- `title`: string. Source article title.
- `patient`: string. Patient note.
- `age`: list of tuples. Each entry is in format `(value, unit)` where value is a float number and unit is in 'year', 'month', 'week', 'day' and 'hour' indicating age unit. For example, `[[1.0, 'year'], [2.0, 'month']]` indicating the patient is a one-year- and two-month-old infant.
- `gender`: 'M' or 'F'. Male or Female.

Patient-patient similarity is stored in `patient2patient_similarity.json`, which is a dict. The keys are `patient_uid` and each value is a tuple of list with entry 0 indicating list of `patient_uid` of notes with similarity 1 and entry 1 indicating similairty 2.

Patient-article relevance is stored in `patient2article_relevance.json`, which is a dict. The keys are `patient_uid` and each value is a list of PMIDs of relevant articles. Note that source article for the patient is included.

## Downstream Tasks
Codes of downstream tasks are in directory `downstream_task/code/`. Datasets for four tasks are in `datasets/`.
### Patient Note Recognition (PNR)
Generate whole PNR dataset (without train/dev/test split) by 
```
python downstream_task/code/dataset_construction/patient_note_recognition.py
```
Train/dev/test split is done by
```
python downstream_task/code/dataset_construction/dataset_split.py
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
python downstream_task/code/dataset_construction/PPS_sample.py
```
PPS dataset is a list of triplets. Each entry is in format (patient_uid_1, patient_uid_2, similarity) where similarity has three values:0, 1, 2, indicating corresponding similarity.

### Patient-Patient Retrieval (PPR)
Split by `dataset_split.py`. PPR dataset is a dict where the keys are `patient_uid` of queries and each entry is two lists of `patient_uid`, representing notes of similarity 1 and 2, respectively.

### Patient-Article Retrieval (PAR)
Split by `dataset_split.py`. PPR dataset is a dict where the keys are `patient_uid` of queries and each entry is a list of `PMID`, representing articles relevant to the query.


## Baseline Models
Codes for build and evaluate baseline models for downstream tasks are in directory `downstream_task/code/baseline`. Evaluation codes are reusable.
To simply reproduce baselines, download xxxxx and unzip it to directory `datasets/`.
### PNR

**Demo_based:**

```
python downstream_task/code/baseline/task_1/demo_filter/demo_baseline.py
```

**BERT:**

`train` argument controls whether to train the model. For other parameters, see code.
```
python downstream_task/code/baseline/task_1/BERT/main.py --train
```
**CNN+LSTM+CRF:**

`train` argument controls whether to train the model. For other parameters, see code.
```
python downstream_task/code/baseline/task_1/CNN_LSTM_CRF/main.py --train
```

### PPS

**Feature_based:**

First use Scispacy (https://github.com/allenai/scispacy) to detect named entities in patient notes and link each entity to UMLS.
```
python downstream_task/code/baseline/task_2/feature_based/NER.py
```
Then train a model with grid search for best hyperparameters.
`model` argument controls whether to use logistic regression or SVM (`LR` or `SVM`).
```
python downstream_task/code/baseline/task_2/feature_based/model.py --model LR
```

**BERT**

`train` argument controls whether to train the model. For other parameters, see code.
```
python downstream_task/code/baseline/task_2/BERT/main.py --train
```

### PPR

**BM25**

Elasticsearch and the python package Elasticsearch is required.
Add index.
```
python downstream_task/code/baseline/task_3/ES/add_index.py
```
Query with multithreads.
```
python downstream_task/code/baseline/task_3/ES/query.py
```

**EBR**

First generate embeddings for each query and document.
```
python downstream_task/code/baseline/task_3/EBR/embedding.py
```
Then perform embedding based retrieval.
```
python downstream_task/code/baseline/task_3/EBR/EBR.py
```


### PAR
Note that to run baseline models for PAR, you have to download PubMed and extract titles and abstracts (see commands above).

**BM25**

Elasticsearch and the python package Elasticsearch is required.
Add index.
```
python downstream_task/code/baseline/task_4/ES/add_index.py
```
Query with multithreads.
```
python downstream_task/code/baseline/task_4/ES/query.py
```

**EBR**

First generate embeddings for each query and document.
```
python downstream_task/code/baseline/task_4/EBR/embedding.py
```
Then perform embedding based retrieval.
```
python downstream_task/code/baseline/task_4/EBR/EBR.py
```

**Co-citation**

```
python downstream_task/code/baseline/task_4/co-citation/co-citation.py
```

