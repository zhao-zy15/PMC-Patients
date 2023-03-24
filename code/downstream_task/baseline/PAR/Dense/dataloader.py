import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np


class PAR_BiEncoder_Dataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 512
        file_name = "PAR_{}_qrels.tsv".format(mode)
        with open(os.path.join(data_dir, file_name), 'r') as f:
            self.instances = [(line.split('\t')[0], line.split('\t')[1]) for line in f.readlines()[1:]]
        patients = json.load(open(os.path.join(data_dir, "../PMC-Patients.json"), "r"))
        self.patients = {patient['patient_uid']: patient for patient in patients}
        corpus_file = "PAR_corpus.jsonl"
        self.corpus = []
        with open(os.path.join(data_dir, corpus_file), "r") as f:
            for line in f:
                self.corpus.append(json.loads(line))
        self.corpus = {article['_id']: {"title": article['title'], "text": article['text']} for article in self.corpus}


    def __getitem__(self, index):
        assert index < len(self.instances)
        patient_uid, PMID = self.instances[index]
        patient = self.patients[patient_uid]['patient']
        patient = self.tokenizer(patient, max_length = self.max_length, padding = "max_length", truncation = True)
        article = self.corpus[PMID]['title'] + self.tokenizer.sep_token + self.corpus[PMID]['text']
        article = self.tokenizer(article, max_length = self.max_length, padding = "max_length", truncation = True)
        return patient["input_ids"], patient["attention_mask"], patient["token_type_ids"], \
            article["input_ids"], article["attention_mask"], article["token_type_ids"]

    
    def __len__(self):
        return len(self.instances)


def MyCollateFn(batch):
    batch = list(zip(*batch))
    input_ids_1 = torch.tensor(batch[0])
    attention_mask_1 = torch.tensor(batch[1])
    token_type_ids_1 = torch.tensor(batch[2])
    input_ids_2 = torch.tensor(batch[3])
    attention_mask_2 = torch.tensor(batch[4])
    token_type_ids_2 = torch.tensor(batch[5])
    return input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2, token_type_ids_2


if __name__ == "__main__":
    data_dir = "../../../../../datasets/patient2article_retrieval"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = PAR_BiEncoder_Dataset(data_dir, "dev", tokenizer)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = MyCollateFn)
    import ipdb; ipdb.set_trace()