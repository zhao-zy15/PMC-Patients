import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np


class PatientSimilarityDataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer, no_train = False):
        self.tokenizer = tokenizer
        self.max_length = 512
        assert mode in ["train", "dev", "test"], "mode argument be one of \'train\', \'dev\' or \'test\'."
        file_name = "PPS_" + mode + ".json"
        self.instances = json.load(open(os.path.join(data_dir, file_name), "r"))
        if no_train:
            case_ids = [ins[0] for ins in self.instances]
            self.instances = list(filter(lambda ins: ins[1] in case_ids, self.instances))
        corpus = json.load(open(os.path.join(data_dir, "../PMC-Patients_" + mode + ".json"), "r"))
        if mode != "train":
            corpus += json.load(open(os.path.join(data_dir, "../PMC-Patients_train.json"), "r"))
        self.corpus = {patient['patient_uid']: patient for patient in corpus}


    def __getitem__(self, index):
        assert index < len(self.instances)
        patient_uid1, patient_uid2, similarity = self.instances[index]
        if similarity == -1:
            similarity = 0
        patient_1 = self.corpus[patient_uid1]['patient']
        patient_2 = self.corpus[patient_uid2]['patient']

        pair = self.tokenizer(patient_1, patient_2, max_length = self.max_length, padding = "max_length", truncation = True)
        return pair["input_ids"], pair["attention_mask"], pair["token_type_ids"], similarity

    
    def __len__(self):
        return len(self.instances)

    
    def stat(self):
        labelCount = [0, 0, 0]
        for _, _, label in self.instances:
            labelCount[label] += 1
        return labelCount


def MyCollateFn(batch):
    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[0])
    attention_mask = torch.tensor(batch[1])
    token_type_ids = torch.tensor(batch[2])
    label = torch.tensor(batch[3])
    return input_ids, attention_mask, token_type_ids, label


if __name__ == "__main__":
    data_dir = "../../../../../datasets/task_2_patient2patient_similarity"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = PatientSimilarityDataset(data_dir, "dev", tokenizer, True)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = MyCollateFn)
    import ipdb; ipdb.set_trace()