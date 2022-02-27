## PMC-Patients Collection
Codes for PMC-Patients collection. Note that downloading PMC OA and pubmed are required (see parent directory).

### Preprocessing
Enumerate all .csv files in PMC OA, filtering those without PMID, and write file_path, PMID, License of each article in a new file.
```
python PMC_OA_utils/PMC_OA_meta.py
```
Extract citations in PMC OA.
```
python PMC_OA_utils/extract_PMC_cites.py
```
Extract citations in PubMed.
```
python pubmed_extractors/extract_cites.py
```
Extract titles and abstracts in PubMed for PAR. (If you only wish to reproduce baseline models, run this command is enough.)
```
python pubmed_extractors/extract_title_abstract.py
```

### Collection
Detect and extract patient notes in PMC OA.
```
python code/extractor.py
```
Filter non-patient-note.
```
python code/filters.py
```
Annotate similarity and relevance.
```
python code/annotate.py
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
