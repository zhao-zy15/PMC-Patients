import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer


class PatientFindingDataset(Dataset):
    def __init__(self, data_dir, tokenizer, mode, max_length):
        assert mode in ["train", "dev", "test", "human"], "mode argument be one of \'train\', \'dev\', \'test\' or \'human\'."
        file_name = "patient_note_recognition_" + mode + ".json"
        articles = json.load(open(os.path.join(data_dir, file_name), "r"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.tags = []
        self.index = []
        for art in articles:
            self.texts += art['texts']
            self.tags += art['tags']
            self.index += [i for i in range(len(art['texts']))]
        self.tag2id = {"B": 1, "I": 2, "O": 0, "E": 2, "S": 1}


    def __getitem__(self, index):
        assert index < self.__len__()
        text = self.texts[index]
        tokens = self.tokenizer(text, max_length = self.max_length, padding = "max_length", truncation = True)
        return tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'], self.tag2id[self.tags[index]], self.index[index]


    def __len__(self):
        return len(self.texts)

    '''
    def summary(self):
        senLen1 = []
        senLen2 = []
        labelCount = [0, 0, 0]

        for sen1, sen2, label in self.data:
            senLen1.append(len(sen1.strip().split()))
            senLen2.append(len(sen2.strip().split()))
            labelCount[label] += 1
        
        import ipdb;ipdb.set_trace()
        return np.mean(senLen1), np.mean(senLen2), labelCount
    '''


def MyCollateFn(batch):
    batch = list(zip(*batch))
    input_ids = torch.tensor(batch[0])
    attention_mask = torch.tensor(batch[1])
    token_type_ids = torch.tensor(batch[2])
    tags = torch.tensor(batch[3])
    index = batch[4]
    return input_ids, attention_mask, token_type_ids, tags, index


if __name__ == "__main__":
    data_dir = "../../../../datasets/task_1_patient_note_recognition"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 512)
    dataset = PatientFindingDataset(data_dir, tokenizer, "human", 512)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = MyCollateFn)
    x = iter(dataloader).next()
    import ipdb; ipdb.set_trace()