# PMC-Patients
PMC-Patients is a first-of-its-kind dataset consisting of 167k patient summaries extracted from case reports in PubMed Central (PMC), 3.1M patient-article relevance and 293k patient-patient similarity annotations defined by PubMed citation graph.
PMC-Patients can serve as a patient collection as well as a benchmark for Retrieval-based Clinical Decision Support (ReCDS) system.
For details, please refer to [our paper](https://www.nature.com/articles/s41597-023-02814-8).

You are free to download the dataset via either [Figshare](https://figshare.com/collections/PMC-Patients/6723465) or [Hugginface](https://huggingface.co/zhengyun21) (including patient summaries and training/dev/test data for ReCDS benchmark) without any data usage agreement. 
For dataset format details, see `README.md` in `datasets` directory.

we also provide a [learderboard](https://pmc-patients.github.io/) for PMC-Patients ReCDS benchmark.


## PMC OA and PubMed Downloads
For those who only wish to reproduce baseline models, only PubMed abstracts are required for PAR task.

If you have already downloaded PMC OA and PubMed abstracts on your device, skip this step and change relative directory in later steps. Otherwise, download [PMC OA](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/) and [PubMed](https://ftp.ncbi.nlm.nih.gov/pubmed/). Note that file `PMC-ids.csv` under [this directory](https://ftp.ncbi.nlm.nih.gov/pub/pmc/) is also required.

## Dataset
After downloading the data, please unzip it and put `datasets` and `meta_data` under this directory.

For dataset details, see `README.md` in `datasets` directory.

All articles used in PMC-Patients are credited in `meta_data/PMC-Patients_citations.json`.

## Code
To reproduce construction of PMC-Patients, see `code/PMC-Patients_collection/`. To try our baselines, see `code/downstream_task/`.

## License
PMC-Patients dataset is released under CC BY-NC-SA 4.0 License.

## Citation
```
@article{Zhao2023ALD,
  title={A large-scale dataset of patient summaries for retrieval-based clinical decision support systems.},
  author={Zhengyun Zhao and Qiao Jin and Fangyuan Chen and Tuorui Peng and Sheng Yu},
  journal={Scientific data},
  year={2023},
  volume={10 1},
  pages={
          909
        },
  url={https://api.semanticscholar.org/CorpusID:266360591}
}
```
