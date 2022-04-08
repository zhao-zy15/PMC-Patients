import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


class PARDataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer, max_length, use_type, pubmed):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_type = use_type
        assert mode in ["train", "dev", "test"], "mode argument be one of \'train\', \'dev\' or \'test\'."
        file_name = "PAR_sample_" + mode + ".json"
        self.instances = []
        samples = json.load(open(os.path.join(data_dir, file_name), "r"))
        for patient in samples:
            for pos in samples[patient]['pos']:
                self.instances.append((patient, pos, 1))
                assert len(pubmed[pos]['title']) > 0 and len(pubmed[pos]['abstract']) > 0
            for neg in samples[patient]['easy_neg']:
                self.instances.append((patient, neg, 0))
                assert len(pubmed[neg]['title']) > 0 and len(pubmed[neg]['abstract']) > 0
            for neg in samples[patient]['hard_neg']:
                self.instances.append((patient, neg, 0))
                assert len(pubmed[neg]['title']) > 0 and len(pubmed[neg]['abstract']) > 0
        
        self.instances = self.instances[0 : (len(self.instances) // 10)]

        patients = json.load(open(os.path.join(data_dir, "../../PMC-Patients_" + mode + ".json"), "r"))
        self.patients = {patient['patient_uid']: patient for patient in patients}
        self.pubmed = pubmed


    def __getitem__(self, index):
        assert index < len(self.instances)
        patient_uid, PMID, label = self.instances[index]
        # For Hinge Loss
        #label = 1 if label == 0 else -1
        patient = self.patients[patient_uid]['patient']
        title = self.pubmed[PMID]['title']
        abstract = self.pubmed[PMID]['abstract']

        title_tokenized = self.tokenizer(patient, title, max_length = self.max_length, padding = "max_length", truncation = True)
        abs_tokenized = self.tokenizer(patient, abstract, max_length = self.max_length, padding = "max_length", truncation = True)

        return title_tokenized["input_ids"], title_tokenized["attention_mask"], title_tokenized["token_type_ids"],\
            abs_tokenized["input_ids"], abs_tokenized["attention_mask"], abs_tokenized["token_type_ids"], label

    
    def __len__(self):
        return len(self.instances)


    def stat(self):
        labels = [0, 0]
        for ins in self.instances:
            labels[ins[-1]] += 1
        return labels


def MyCollateFn(batch):
    batch = list(zip(*batch))
    t_input_ids = torch.tensor(batch[0])
    t_attention_mask = torch.tensor(batch[1])
    t_token_type_ids = torch.tensor(batch[2])
    a_input_ids = torch.tensor(batch[3])
    a_attention_mask = torch.tensor(batch[4])
    a_token_type_ids = torch.tensor(batch[5])
    label = torch.tensor(batch[6])
    return t_input_ids, t_attention_mask, t_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids, label


if __name__ == "__main__":
    data_dir = "../../../../../datasets/task_4_patient2article_retrieval/PAR_sample"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pubmed = {}
    pubmed_dir = "../../../../../../pubmed/pubmed_title_abstract/"
    for file_name in tqdm(os.listdir(pubmed_dir)):
        articles = json.load(open(os.path.join(pubmed_dir, file_name), "r"))
        for PMID in articles:
            if PMID not in pubmed or len(pubmed[PMID]['title']) * len(pubmed[PMID]['abstract']) == 0:
                pubmed[PMID] = articles[PMID]
    dataset = PARDataset(data_dir, "train", tokenizer, 512, "abstract", pubmed)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, collate_fn = MyCollateFn)
    x = iter(dataloader).next()
    import ipdb; ipdb.set_trace()
