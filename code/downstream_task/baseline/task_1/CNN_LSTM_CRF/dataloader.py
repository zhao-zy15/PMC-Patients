import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.corpus import stopwords as pw
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


class PatientFindingDataset(Dataset):
    def __init__(self, tokenizer, max_length, data_dir, mode):
        assert mode in ["train", "dev", "test", "human"], "mode argument be one of \'train\', \'dev\', \'test\' or \'human\'."
        # Use human annotations as golden labels.
        if mode == "human":
            mode = "human_annotations"
        file_name = "PNR_" + mode + ".json"
        self.stopword = pw.words('english')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instances = json.load(open(os.path.join(data_dir, file_name), "r"))
        self.tag2id = {"<SOS>": 0, "B": 1, "I": 2, "O": 3, "<EOS>": 4}
        


    def __getitem__(self, index):
        assert index < len(self.instances)
        texts = self.instances[index]['texts']
        input_ids = []
        
        for text in texts:
            text = ' '.join(list(filter(lambda x: x not in self.stopword, text.split())))
            tokens = self.tokenizer(text, max_length = self.max_length, padding = "max_length", truncation = True)
            input_ids.append(tokens['input_ids'])
        tags = [self.tag2id[tag] for tag in self.instances[index]['tags']]
        
        return torch.tensor(input_ids), torch.tensor(tags)


    def __len__(self):
        return len(self.instances)


def MyCollateFn(batch):
    input_ids = [x[0] for x in batch]
    input_ids = torch.cat(input_ids, 0)
    tags =  [x[1] for x in batch]
    length = [len(x[1]) for x in batch]
    tags = pad_sequence(tags, batch_first = True, padding_value = -1)
    return input_ids, tags, length


if __name__ == "__main__":
    data_dir = "../../../../../datasets/task_1_patient_note_recognition"
    
    file_name = "PNR_test.json"
    instances = json.load(open(os.path.join(data_dir, file_name), "r"))
    length = []
    st = pw.words('english')
    for ins in instances:
        for text in ins['texts']:
            length.append(len(list(filter(lambda x: x not in st, text.split())) ))
    import ipdb; ipdb.set_trace()
    
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 512)
    dataset = PatientFindingDataset(tokenizer, 256, data_dir, "test")
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = MyCollateFn)
    x = iter(dataloader).next()
    import ipdb; ipdb.set_trace()