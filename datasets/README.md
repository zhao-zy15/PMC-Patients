## PMC-Patients Dataset
PMC-Patients dataset can be download via [this link](https://drive.google.com/file/d/1vFCLy_CF8fxPDZvDtHPR6Dl6x9l0TyvW/view?usp=sharing) without any data usage agreement. Dataset consists of the following files (all files listed below have train, dev, and test splits).

### PMC-Paitents_train.json

PMC-Patients-training/test/dev, each of which is a list of dict with keys:
- `patient_id`: string. A continuous id of patients, starting from 0.
- `patient_uid`: string. Unique ID for each patient, with format PMID-x, where PMID is the PubMed Identifier of source article of the note and x denotes index of the note in source article.
- `PMID`: string. PMID for source article.
- `file_path`: string. File path of xml file of source article.
- `title`: string. Source article title.
- `patient`: string. Patient note.
- `age`: list of tuples. Each entry is in format `(value, unit)` where value is a float number and unit is in 'year', 'month', 'week', 'day' and 'hour' indicating age unit. For example, `[[1.0, 'year'], [2.0, 'month']]` indicating the patient is a one-year- and two-month-old infant.
- `gender`: 'M' or 'F'. Male or Female.

Note that `PMC-Patients_human.json` present ground-truth patient notes, their ages, and genders. The `human_patient_id` and `human_patient_uid` are different from those in `PMC-Patients` since it's difficult to align human annotations with automatic annotations.

### task_1_patient_note_recognition/PNR_train.json

List of dict with keys:
- `PMID`: string. PMID of the article.
- `file_path`: string. File path of xml in PMC OA.
- `title`, `abstract`: string. Title and abstract of the article.
- `texts`: list of string. Each entry is one paragraph of the article.
- `tags`: list of string. BIOES tags. Each entry indicates tag for corresponding paragraph.

### task_2_patient2patient_similarity/PPS_train.json

PPS dataset is a list of triplets. Each entry is in format (patient_uid_1, patient_uid_2, similarity) where similarity has three values:0, 1, 2, indicating corresponding similarity.

### task_3_patient2patient_retrieval/PPR_train.json

A dict where the keys are `patient_uid` of queries and each entry is two lists of `patient_uid`, representing notes of similarity 1 and 2, respectively.

### task_4_patient2article_retrieval/PAR_train.json

A dict where the keys are `patient_uid` of queries and each entry is a list of `PMID`, representing articles relevant to the query.

### PPR_PAR_human_annotations.json

Ground-truth of patient-patient similarity and patient-article relevance of top 5 results given by BM25 on human-annotated patient notes.

The keys are `human_patient_id` and each entry is a dict with `PMID`s and `patient_uid`s as keys, representing articles and patients, respectively, and numbers as values indicating type of relevance/similarity. For details, see our paper.