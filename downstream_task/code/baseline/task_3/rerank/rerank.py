import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
sys.path.append("../../task_2/BERT")
from model import monoBERT
sys.path.append("..")
from utils import getPrecision, getRR, getRecall


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


# PPS model output_dir.
output_dir = "../../task_2/BERT/output_56_hard"
model_path = os.path.join(output_dir, "best_model.pth")
args = torch.load(os.path.join(output_dir, "training_args.bin"))
model_name_or_path = args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 512)
patients = json.load(open("../../../../../meta_data/PMC-Patients.json", "r"))
patient_text = {patient['patient_uid']: patient['patient'] for patient in patients}
batch_size = 512
max_length = 512
device = "cuda:0"
# Candidates given by Elasticsearch.
candidates = json.load(open("../patient2patient_retrieved_test.json"))

mse = args.mse
model = monoBERT(model_name_or_path, True)
model = torch.load(model_path)
model.to(device)
model.eval()

PPS = {}
input_ids = []
attention_mask = []
token_type_ids = []
pairs = []
for patient in tqdm(candidates):
    # Only rerank top 1k candidates due to time cost.
    for candidate in candidates[patient][:1000]:
        if (candidate, patient) not in PPS:
            tokenized = tokenizer(patient_text[patient], patient_text[candidate], max_length = max_length, padding = "max_length", truncation = True)
            pairs.append((patient, candidate))
            input_ids.append(tokenized["input_ids"])
            attention_mask.append(tokenized["attention_mask"])
            token_type_ids.append(tokenized["token_type_ids"])
            if len(input_ids) == batch_size:
                input_ids = torch.tensor(input_ids).to(device)
                attention_mask = torch.tensor(attention_mask).to(device)
                token_type_ids = torch.tensor(token_type_ids).to(device)
                with torch.no_grad():
                    _, score = model(input_ids, attention_mask, token_type_ids)
                if mse:
                    score = score.detach().cpu()
                    for i in range(len(pairs)):
                        PPS[pairs[i]] = score[i].item()
                else:
                    prob = F.softmax(score, dim = 1)
                    prob = prob.detach().cpu().numpy()
                    for i in range(len(pairs)):
                        PPS[pairs[i]] = prob[i].tolist()

                input_ids = []
                attention_mask = []
                token_type_ids = []
                pairs = []

# Remaning samples.
if input_ids:
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    with torch.no_grad():
        _, score = model(input_ids, attention_mask, token_type_ids)
    if mse:
        score = score.detach().cpu()
        for i in range(len(pairs)):
            PPS[pairs[i]] = score[i].item()
    else:
        prob = F.softmax(score, dim = 1)
        prob = prob.detach().cpu().numpy()
        for i in range(len(pairs)):
            PPS[pairs[i]] = prob[i].tolist()


# Cache rerank scores.
#json.dump({' '.join(k): v for k,v in PPS.items()}, open("PPS.json", "w"), indent = 4)


#PPS = json.load(open("PPS.json", "r"))
RR = []
precision = []
recall_100 = []
recall = []
# Predicted number of each label.
predict = [0, 0, 0]
# Not used for MSE loss. For CE loss, weights for three probabilities of labels.
weight = [0, 1, 2]
dataset = json.load(open("../../../../../datasets/task_3_patient2patient_retrieval/PPR_test.json"))
for patient in tqdm(candidates):
    temp = []
    for candidate in candidates[patient][:1000]:
        if ' '.join((patient, candidate)) in PPS:
            temp.append((candidate, PPS[' '.join((patient, candidate))]))
        else:
            temp.append((candidate, PPS[' '.join((candidate, patient))]))
    if mse:
        temp = sorted(temp, key = lambda x: x[1], reverse = True)
        for item in temp:
            if item < 0.5:
                predict[0] += 1
            elif item < 1.5:
                predict[1] += 1
            else:
                predict[2] += 1
    else:
        temp = sorted(temp, key = lambda x: np.dot(x[1], weight), reverse = True)
        for item in temp:
            predict[np.argmax(item[1])] += 1
    result_ids = [x[0] for x in temp]
    golden_labels = dataset[patient][0] + dataset[patient][1]
    RR.append(getRR(golden_labels, result_ids))
    precision.append(getPrecision(golden_labels, result_ids[:10]))
    recall_100.append(getRecall(golden_labels, result_ids[:100]))
    recall.append(getRecall(golden_labels, result_ids))

# Average predicted number of each label.
print(np.array(predict) / len(RR))
print(np.mean(RR), np.mean(precision), np.mean(recall_100), np.mean(recall))
print(len(RR))
    

