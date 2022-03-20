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
        file_name = "PAR_" + mode + ".json"
        instances = json.load(open(os.path.join(data_dir, file_name), "r"))
        self.rel = instances
        self.instances = [(k, v_i) for k,v in instances.items() for v_i in v]
        patients = json.load(open(os.path.join(data_dir, "../PMC-Patients_" + mode + ".json"), "r"))
        self.patients = {patient['patient_uid']: patient for patient in patients}
        self.pubmed = pubmed
        '''
        self.citations = {}
        pubmed_citation_dir = "../../../../../../../pubmed/pubmed_citations"
        for file_name in tqdm(os.listdir(pubmed_citation_dir)):
            cites = json.load(open(os.path.join(pubmed_citation_dir, file_name), "r"))
            self.citations.update(cites)
        '''


    def __getitem__(self, index):
        assert index < len(self.instances)
        patient_uid, PMID = self.instances[index]
        patient = self.patients[patient_uid]['patient']
        title = self.pubmed[PMID]['title']
        abstract = self.pubmed[PMID]['abstract']
        article = (title + '\n' + abstract).strip()

        # TODO:use_type == both
        tokenized_1 = self.tokenizer(patient, max_length = self.max_length[0], padding = "max_length", truncation = True)
        #max_length_2 = max(self.max_length[1], sum(self.max_length) - len(tokenized_1['input_ids']))
        tokenized_2 = self.tokenizer(article, max_length = self.max_length[1] + 1, padding = "max_length", truncation = True)
        
        # For hard negative sampling
        '''
        candidates = set(self.citations[PMID]) if PMID in self.citations else set()
        candidates = candidates - set(self.rel[patient_uid])
        candidates = [(self.pubmed[x]['title'] + '\n' + self.pubmed[x]['abstract']).strip() for x in list(candidates)]
        '''
        candidates = None
        return tokenized_1["input_ids"] + tokenized_2["input_ids"][1:], tokenized_1["attention_mask"] + tokenized_2["attention_mask"][1:], \
            tokenized_1["token_type_ids"] + [1] * (len(tokenized_2['input_ids']) - 1), PMID, self.rel[patient_uid], candidates

    
    def __len__(self):
        return len(self.instances)


def MyCollateFn(batch, max_length, neg_ratio, tokenizer):
    batch = list(zip(*batch))
    batch[0] = list(batch[0])
    batch[1] = list(batch[1])
    batch[2] = list(batch[2])
    PMIDs = batch[3]
    PMID2id = {PMIDs[i]: i for i in range(len(batch[3]))}
    # TODO: ratio of pos:neg
    neg_samples = {}
    hard_neg_samples = {}
    labels = []
    for i in range(len(batch[0])):
        labels.append(1)
        neg_samples[i] = []
        permutation = np.random.permutation(PMIDs)
        for PMID in permutation:
            if PMID in batch[-1][i]:
                continue
            else:
                neg_samples[i].append(PMID2id[PMID])
                if len(neg_samples[i]) == neg_ratio:
                    break
        #if len(batch[-1][i]) > 0:
        #    hard_neg_samples[i] = np.random.choice(batch[-1][i], 1)[0]

    for x in neg_samples:
        for y in neg_samples[x]:
            batch[0].append(batch[0][x][:max_length[0]] + batch[0][y][max_length[0]:])
            batch[1].append(batch[1][x][:max_length[0]] + batch[1][y][max_length[0]:])
            batch[2].append(batch[2][x][:max_length[0]] + batch[2][y][max_length[0]:])
            labels.append(0)
        if x in hard_neg_samples:
            tokenized = tokenizer(hard_neg_samples[x], max_length = max_length[1] + 1, padding = "max_length", truncation = True)
            batch[0].append(batch[0][x][:max_length[0]] + tokenized['input_ids'][1:])
            batch[1].append(batch[1][x][:max_length[0]] + tokenized['attention_mask'][1:])
            batch[2].append(batch[2][x][:max_length[0]] + [1] * (len(tokenized['input_ids']) - 1))
            labels.append(0)

    input_ids = torch.tensor(batch[0])
    attention_mask = torch.tensor(batch[1])
    token_type_ids = torch.tensor(batch[2])
    label = torch.tensor(labels)
    return input_ids, attention_mask, token_type_ids, label


if __name__ == "__main__":
    data_dir = "../../../../../datasets/task_4_patient2article_retrieval"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pubmed = {}
    pubmed_dir = "../../../../../../../pubmed/pubmed_title_abstract/"
    for file_name in tqdm(os.listdir(pubmed_dir)):
        articles = json.load(open(os.path.join(pubmed_dir, file_name), "r"))
        for PMID in articles:
            pubmed[PMID] = articles[PMID]
    dataset = PARDataset(data_dir, "test", tokenizer, [128, 384], "abstract", pubmed)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, collate_fn = lambda x: MyCollateFn(x, [128, 384], 1, tokenizer))
    x = iter(dataloader).next()
    import ipdb; ipdb.set_trace()
