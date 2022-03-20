import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import monoBERT
import sys
sys.path.append("..")
from utils import getPrecision, getRR, getRecall


# PAR model output_dir.
output_dir = "output"
model_path = os.path.join(output_dir, "best_model.pth")
args = torch.load(os.path.join(output_dir, "training_args.bin"))
model_name_or_path = args.model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 512)
patients = json.load(open("../../../../../meta_data/PMC-Patients.json", "r"))
patient_text = {patient['patient_uid']: patient['patient'] for patient in patients}
'''
pubmed = {}
data_dir = "../../../../../../pubmed/pubmed_title_abstract/"
for file_name in tqdm(os.listdir(data_dir)):
    articles = json.load(open(os.path.join(data_dir, file_name), "r"))
    for PMID in articles:
        pubmed[PMID] = articles[PMID]
'''
batch_size = 512
device = "cuda:1"

# Candidates given by KNN.
candidates = json.load(open("../knn_patient2article_retrieved_test.json"))
'''
model = monoBERT(model_name_or_path)
model = torch.load(model_path)
model.to(device)
model.eval()

PAR = {}
input_ids = []
attention_mask = []
token_type_ids = []
pairs = []
for patient in tqdm(candidates):
    # Only rerank top 1k candidates due to time cost.
    for candidate in candidates[patient][:100]:
        pairs.append((patient, candidate))
        title = pubmed[candidate]['title']
        abstract = pubmed[candidate]['abstract']
        article = (title + abstract).strip()
        tokenized_1 = tokenizer(patient_text[patient], max_length = args.max_length[0], padding = "max_length", truncation = True)
        tokenized_2 = tokenizer(article, max_length = args.max_length[1] + 1, padding = "max_length", truncation = True)
        
        input_ids.append(tokenized_1["input_ids"] + tokenized_2["input_ids"][1:])
        attention_mask.append(tokenized_1["attention_mask"] + tokenized_2["attention_mask"][1:])
        token_type_ids.append(tokenized_1["token_type_ids"] + [1] * (len(tokenized_2['input_ids']) - 1))
        if len(input_ids) == batch_size:
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            token_type_ids = torch.tensor(token_type_ids).to(device)
            with torch.no_grad():
                _, score = model(input_ids, attention_mask, token_type_ids)
                score = score.detach().cpu()
                for i in range(len(pairs)):
                    PAR[pairs[i]] = score[i].item()

            input_ids = []
            attention_mask = []
            token_type_ids = []
            pairs = []

# Remaining samples.
if input_ids:
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    with torch.no_grad():
        _, score = model(input_ids, attention_mask, token_type_ids)
        score = score.detach().cpu()
        for i in range(len(pairs)):
            PAR[pairs[i]] = score[i].item()


# Cache rerank scores.
json.dump({' '.join(k): v for k,v in PAR.items()}, open("PAR.json", "w"), indent = 4)
'''

PAR = json.load(open("PAR.json", "r"))
RR = []
precision = []
recall_100 = []
recall = []
# Predicted number of each label.
predict = [0, 0]
dataset = json.load(open("../../../../../datasets/task_4_patient2article_retrieval/PAR_test.json"))
for patient in tqdm(candidates):
    temp = []
    for candidate in candidates[patient][:100]:
        if ' '.join((patient, candidate)) in PAR:
            temp.append((candidate, PAR[' '.join((patient, candidate))]))
        else:
            temp.append((candidate, PAR[' '.join((candidate, patient))]))
    temp = sorted(temp, key = lambda x: x[1], reverse = True)
    for item in temp:
        if item[1] < 0.5:
            predict[0] += 1
        else:
            predict[1] += 1
    result_ids = [x[0] for x in temp]
    golden_labels = dataset[patient]
    RR.append(getRR(golden_labels, result_ids))
    precision.append(getPrecision(golden_labels, result_ids[:10]))
    recall_100.append(getRecall(golden_labels, result_ids[:100]))
    recall.append(getRecall(golden_labels, result_ids))

# Average predicted number of each label.
print(np.array(predict) / len(RR))
print(np.mean(RR), np.mean(precision), np.mean(recall_100), np.mean(recall))
print(len(RR))
    

