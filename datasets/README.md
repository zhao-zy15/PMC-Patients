# Dataset

PMC-Patients dataset can be download via [Figshare](https://figshare.com/collections/PMC-Patients/6723465) or [Hugginface](https://huggingface.co/zhengyun21) without any data usage agreement. Dataset consists of the following files (all files listed below have train, dev, and test splits).

We present our data in three parts: PMC-Patients Dataset, PMC-Patients ReCDS Benchmark, and PMC-Patients Meta Data.

## PMC-Patients Dataset

The core file of our dataset, containing the patient summaries, demographics, and relational annotations.

### PMC-Patients.json
Patient summaries are presented as a `json` file, which is a list of dictionaries with the following keys:
- `patient_id`: string. A continuous id of patients, starting from 0.
- `patient_uid`: string. Unique ID for each patient, with format PMID-x, where PMID is the PubMed Identifier of source article of the note and x denotes index of the note in source article.
- `PMID`: string. PMID for source article.
- `file_path`: string. File path of xml file of source article.
- `title`: string. Source article title.
- `patient`: string. Patient note.
- `age`: list of tuples. Each entry is in format `(value, unit)` where value is a float number and unit is in 'year', 'month', 'week', 'day' and 'hour' indicating age unit. For example, `[[1.0, 'year'], [2.0, 'month']]` indicating the patient is a one-year- and two-month-old infant.
- `gender`: 'M' or 'F'. Male or Female.
- `relevant_articles`: dict. The key is PMID of the relevant articles and the corresponding value is its relevance score (2 or 1 as defined in the ``Methods'' section).
- `similar_patients`: dict. The key is patient_uid of the similar patients and the corresponding value is its similarity score (2 or 1 as defined in the ``Methods'' section).


## PMC-Patients ReCDS Benchmark

The PMC-Patients ReCDS benchmark is presented as retrieval tasks and the data format is the same as [BEIR](https://github.com/beir-cellar/beir) benchmark. 
To be specific, there are queries, corpus, and qrels (annotations).

### Queries

ReCDS-PAR and ReCDS-PPR tasks share the same query patient set and dataset split.
For each split (train, dev, and test), queries are stored a `jsonl` file that contains a list of dictionaries, each with two fields: 
- `_id`: unique query identifier represented by patient_uid.
- `text`: query text represented by patient summary text.

### Corpus

Corpus is shared by different splits. For ReCDS-PAR, the corpus contains 11.7M PubMed articles, and for ReCDS-PPR, the corpus contains 155.2k reference patients from PMC-Patients. The corpus is also presented by a `jsonl` file that contains a list of dictionaries with three fields:
- `_id`:  unique document identifier represented by PMID of the PubMed article in ReCDS-PAR, and patient_uid of the candidate patient in ReCDS-PPR.
- `title`: : title of the article in ReCDS-PAR, and empty string in ReCDS-PPR.
- `text`: abstract of the article in ReCDS-PAR, and patient summary text in ReCDS-PPR.

### Qrels

Qrels are TREC-style retrieval annotation files in `tsv` format.
A qrels file contains three tab-separated columns, i.e. the query identifier, corpus identifier, and score in this order. The scores (2 or 1) indicate the relevance level in ReCDS-PAR or similarity level in ReCDS-PPR.

Note that the qrels may not be the same as `relevant_articles` and `similar_patients` in `PMC-Patients.json` due to dataset split (see our manuscript for details).


## PMC-Patients Meta Data

Meta data for PMC-Patients that might facilitate reproduction or usage of our dataset, consisting of the following files (most of which can be derived from our main files above).

### PMIDs.json

PMIDs of articles from which PMC-Patients are extracted.
List of string, length 140,897.

### train_PMIDs.json & dev_PMIDs.json & test_PMIDs.json

PMIDs of articles in training / dev / test split.
List of string.

### train_patient_uids.json & dev_patient_uids.json & test_patient_uids.json

Patient_uids of notes in training / dev / test split.
List of string.

### patient2article_relevance.json

Full patient-to-article dataset.
A dict where the keys are `patient_uid` of queries and each entry is a list of `PMID`, representing articles relevant to the query.

The 3-point relevance can be obtained by checking whether the `PMID` is in `PMIDs.json`.

### patient2patient_similarity.json

Full patient-to-patient similarity dataset.
A dict where the keys are `patient_uid` of queries and each entry is a list of `patient_uid`, representing similar patients to the query.

The 3-point similarity can be obtained by checking whether the similar patient share the `PMID` (the string before '-' in `patient_uid`) with the query patient.


### PMID2Mesh.json

Dict of PMIDs to MeSH terms of the article.

### MeSH_Humans_patient_uids.json

`patient_uid` of the patients in PMC-Patients-Humans (extracted from articles with "Humans" MeSH term).
List of string.

### PMC-Patients_citations.json

Citations for all articles we used to collect our dataset.
A dict where the keys are `patient_uid` and each entry is the citation of the source article.

### human_PMIDs.json

PMIDs of the 500 randomly sampled articles for human evaluation.
List of string.

### PMC-Patients_human_eval.json

Expert annotation results of the 500 articles in `human_PMIDs.json`, including manually annotated patient note, demographics, and relations of the top 5 retrieved articles / patients.
List of dict, and the keys are almost identical to `PMC-Patients.json`, with the exception of `human_patient_id` and `human_patient_uid`.

The relational annotations are different from automatic ones. They are strings indicating on which dimension(s) are the patient-article / patient-patient pair relevant / similar. 
"0", "1", "2", and "3" represent "Irrelevant", "Diagnosis", "Test", "Treatment" in ReCDS-PAR, and represent "Dissimilar", "Features", "Outcomes", "Exposure" in ReCDS-PPR.
Note that a pair can be relevant / similar on multiple dimensions at the same time.

### PAR_PMIDs.json

PMIDs of the 11.7M articles used as PAR corpus.
List of string.

