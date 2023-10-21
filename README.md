# PMC-Patients
PMC-Patients is a first-of-its-kind dataset consisting of 167k patient summaries extracted from case reports in PubMed Central (PMC), 3.1M patient-article relevance and 293k patient-patient similarity annotations defined by PubMed citation graph.
PMC-Patients can serve as a patient collection as well as a benchmark for Retrieval-based Clinical Decision Support (ReCDS) system.
For details, please refer to [our paper](https://arxiv.org/pdf/2202.13876.pdf).

You are free to [download the dataset](https://drive.google.com/file/d/1BnZT3MaOry-92Kj4pzC7u1bFYCpNn0ct/view?usp=sharing) (including patient summaries and training/dev/test data for ReCDS benchmark) without any data usage agreement. 
For dataset format details, see `README.md` in `datasets` and `meta_data` directory.

we also provide a [learderboard]() for PMC-Patients ReCDS benchmark.


## PMC OA and PubMed Downloads
For those who only wish to reproduce baseline models, only PubMed abstracts are required for PAR task.

If you have already downloaded PMC OA and PubMed abstracts on your device, skip this step and change relative directory in later steps. Otherwise, download [PMC OA](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/) and [PubMed](https://ftp.ncbi.nlm.nih.gov/pubmed/). Note that file `PMC-ids.csv` under [this directory](https://ftp.ncbi.nlm.nih.gov/pub/pmc/) is also required.

## Dataset
PMC-Patients dataset can be downloaded via [this link](https://drive.google.com/file/d/1BnZT3MaOry-92Kj4pzC7u1bFYCpNn0ct/view?usp=sharing) without any data usage agreement. After downloading, please unzip it and put `datasets` and `meta_data` under this directory.

For dataset details, see `README.md` in `datasets` and `meta_data` directory.

All articles used in PMC-Patients are credited in `meta_data/PMC-Patients_citations.json`.

### Dataset Version Logs
Generally **PMC-Patients** will only be updated incrementally when new data are ready to release, so there's no need to keep an old version and the download link would stay the same.

- `v1.3`: Add `PPR_PAR_human_annotations.json` of ground-truth patient-patient similarity and patient-article relevance.
- `v1.2`: Add `PMC-Patients_human.json` of ground-truth patient notes and their demographics annotated by experts.
- `v1.1`: Add citations of articles used in **PMC-Patients**.

## Code
To reproduce construction of PMC-Patients, see `code/PMC-Patients_collection/`. To try our baselines, see `code/downstream_task/`.

## License
PMC-Patients dataset is released under CC BY-NC-SA 4.0 License.

## Citation
```
@misc{zhao2022pmcpatients,
      title={PMC-Patients: A Large-scale Dataset of Patient Notes and Relations Extracted from Case Reports in PubMed Central}, 
      author={Zhengyun Zhao and Qiao Jin and Sheng Yu},
      year={2022},
      eprint={2202.13876},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```