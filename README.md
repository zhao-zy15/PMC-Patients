# PMC-Patients
This repository contains PMC-Patients dataset (including patient notes, patient-patient similarity annotations, patient-article relevance annotations and four downstream task datasets: patient note recognition PNR, patient-patient similarity PPS, patient-patient retrieval PPR, patient-article retrieval PAR), codes for collection datasets and several baseline models.

See our paper at xxx.

## PMC OA and PubMed Downloads
For those who only wish to reproduce baseline models, only PubMed abstracts are required for PAR task.
If you have already downloaded PMC OA and PubMed abstracts on your device, skip this step and change relative directory in later steps. Otherwise, download PMC OA and PubMed via https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/ and https://ftp.ncbi.nlm.nih.gov/pubmed/, respectively. Note that file `PMC-ids.csv` under directory https://ftp.ncbi.nlm.nih.gov/pub/pmc/ is also required.

## Dataset
Please download PMC-Patients dataset via xxxx (unless you wish to reproduce your own dataset), and unzip it under current directory.

## Code
To reproduce construction of PMC-Patients, see `code/PMC-Patients_collection/`. To try our baselines, see `code/downstream_task/`.

## License
PMC-Patients dataset is released under CC-BY-NC-SA 4.0 License.

## Cite

