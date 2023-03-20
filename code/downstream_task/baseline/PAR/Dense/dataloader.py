import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np


class PPR_BiEncoder_Dataset(Dataset):
    def __init__(self, data_dir, mode, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 512
        file_name = "PPR_{}.json".format(mode)
        data = json.load(open(os.path.join(data_dir, file_name), "r"))
        self.instances = []
        for query in data:
            for sim in data[query]:
                self.instances.append((query, sim))
        patients = json.load(open(os.path.join(data_dir, "../../meta_data/PMC-Patients.json"), "r"))
        self.patients = {patient['patient_uid']: patient for patient in patients}


    def __getitem__(self, index):
        assert index < len(self.instances)
        patient_uid1, patient_uid2 = self.instances[index]
        patient_1 = self.patients[patient_uid1]['patient']
        patient_2 = self.patients[patient_uid2]['patient']
        patient_1 = self.tokenizer(patient_1, max_length = self.max_length, padding = "max_length", truncation = True)
        patient_2 = self.tokenizer(patient_2, max_length = self.max_length, padding = "max_length", truncation = True)
        return patient_1["input_ids"], patient_1["attention_mask"], patient_1["token_type_ids"], \
            patient_2["input_ids"], patient_2["attention_mask"], patient_2["token_type_ids"]

    
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
    data_dir = "../../../../../datasets/patient2patient_retrieval"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = PPR_BiEncoder_Dataset(data_dir, "dev", tokenizer)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = MyCollateFn)
    import ipdb; ipdb.set_trace()