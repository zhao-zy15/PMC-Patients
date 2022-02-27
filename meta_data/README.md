Directory for meta data of PMC-Patients. Consisting of the following files.

`PMIDs.json`

PMIDs of articles from which PMC-Patients are extracted.
List of string, length 140,987 in V1.0.

`train/dev/test_PMIDs.json`

PMIDs of articles in PMC-Patients-training/dev/test.
List of string.

`train/dev/tes_patient_uids.json`

Patient_uids of notes in PMC-Patients-training/dev/test.
List of string.

`PMC-Patients.json`

PMC-Patients dataset, which is a list of dict with keys:
- `patient_id`: string. A continuous id of patients, starting from 0.
- `patient_uid`: string. Unique ID for each patient, with format PMID-x, where PMID is the PubMed Identifier of source article of the note and x denotes index of the note in source article.
- `PMID`: string. PMID for source article.
- `file_path`: string. File path of xml file of source article.
- `title`: string. Source article title.
- `patient`: string. Patient note.
- `age`: list of tuples. Each entry is in format `(value, unit)` where value is a float number and unit is in 'year', 'month', 'week', 'day' and 'hour' indicating age unit. For example, `[[1.0, 'year'], [2.0, 'month']]` indicating the patient is a one-year- and two-month-old infant.
- `gender`: 'M' or 'F'. Male or Female.

`patient_note_recognition.json`

Full PNR dataset.
List of dict with keys:
- `PMID`: string. PMID of the article.
- `file_path`: string. File path of xml in PMC OA.
- `title`, `abstract`: string. Title and abstract of the article.
- `texts`: list of string. Each entry is one paragraph of the article.
- `tags`: list of string. BIOES tags. Each entry indicates tag for corresponding paragraph.

`patient2patient_similarity.json`

Full PPR dataset.
A dict where the keys are `patient_uid` of queries and each entry is two lists of `patient_uid`, representing notes of similarity 1 and 2, respectively.

`patient2article_relevance.json`

Full PAR dataset.
A dict where the keys are `patient_uid` of queries and each entry is a list of `PMID`, representing articles relevant to the query.

`PMID2keywords/Mesh.json`

Dict of PMIDs to keywords/Mesh terms of the article.

`PPS_graph.json`

Graph structure of PPS dataset.
Dict where the keys are `patient_uid` and each entry is $x$ list and list $i$ presents patient_uids at hop $i$.