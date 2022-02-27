## PMC-Patients Dataset
Consisting of the following files.

`PMC-Paitents_train/test/dev.json`

PMC-Patients-training/test/dev.

`task_1_patient_note_recognition/PNR_train/dev/test.json`

List of dict with keys:
- `PMID`: string. PMID of the article.
- `file_path`: string. File path of xml in PMC OA.
- `title`, `abstract`: string. Title and abstract of the article.
- `texts`: list of string. Each entry is one paragraph of the article.
- `tags`: list of string. BIOES tags. Each entry indicates tag for corresponding paragraph.

`task_2_patient2patient_similarity/PPS_train/dev/test.json`

PPS dataset is a list of triplets. Each entry is in format (patient_uid_1, patient_uid_2, similarity) where similarity has three values:0, 1, 2, indicating corresponding similarity.

`task_3_patient2patient_retrieval/PPR_train/dev/test.json`

A dict where the keys are `patient_uid` of queries and each entry is two lists of `patient_uid`, representing notes of similarity 1 and 2, respectively.

`task_4_patient2article_retrieval/PAR_train/dev/test.json`

A dict where the keys are `patient_uid` of queries and each entry is a list of `PMID`, representing articles relevant to the query.
