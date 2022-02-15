import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, tokenizer, patient_dir):
        self.tokenizer = tokenizer
        self.max_length = 512
        patients = json.load(open(patient_dir, "r"))
        self.patients = [patient['patient'] for patient in patients]


    def __getitem__(self, index):
        assert index < len(self.patients)
        tokenized = self.tokenizer(self.patients[index], max_length = self.max_length, padding = "max_length", truncation = True)
        return tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"]

    
    def __len__(self):
        return len(self.patients)


def MyCollateFn(batch):
    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[0])
    attention_mask = torch.tensor(batch[1])
    token_type_ids = torch.tensor(batch[2])
    return input_ids, attention_mask, token_type_ids


model_name_or_path = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 512)
dataset = MyDataset(tokenizer, "../../../../meta_data/PMC-Patients.json")
batch_size = 512
device = "cuda:0"
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn = MyCollateFn)

model = AutoModel.from_pretrained(model_name_or_path)
model.to(device)
model.eval()

def cal_dist(id1, id2):
    input_ids = torch.tensor(dataset[id1][0]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(dataset[id1][1]).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(dataset[id1][2]).unsqueeze(0).to(device)
    x = model(input_ids, attention_mask, token_type_ids)[1].detach().numpy()
    x = (x / np.linalg.norm(x)).reshape(-1)

    input_ids = torch.tensor(dataset[id2][0]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(dataset[id2][1]).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(dataset[id2][2]).unsqueeze(0).to(device)
    y = model(input_ids, attention_mask, token_type_ids)[1].detach().numpy()
    y = (y / np.linalg.norm(y)).reshape(-1)
    return x, y, x.dot(y)

import ipdb; ipdb.set_trace()

embeddings = None
bar = tqdm(dataloader)
for _, batch in enumerate(bar):
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    token_type_ids = batch[2].to(device)
    with torch.no_grad():
        pooled = model(input_ids, attention_mask, token_type_ids)[1].detach().cpu().numpy()
    if embeddings is not None:
        embeddings = np.concatenate((embeddings, pooled), axis = 0)
    else:
        embeddings = pooled

np.save("embeddings.npy", embeddings)