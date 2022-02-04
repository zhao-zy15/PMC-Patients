import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class PatientFindingDataset(Dataset):
    def __init__(self, data_dir, mode):
        assert mode in ["train", "dev", "test", "human"], "mode argument be one of \'train\', \'dev\', \'test\' or \'human\'."
        file_name = "patient_note_recognition_" + mode + ".json"
        self.instances = json.load(open(os.path.join(data_dir, file_name), "r"))
        
        self.tag2id = {"<SOS>": 0, "B": 1, "I": 2, "O": 3, "E": 4, "S": 5, "<EOS>": 6}


    def __getitem__(self, index):
        assert index < len(self.instances)
        PMID = self.instances[index]['PMID']
        texts = self.instances[index]['texts']
        tags = [self.tag2id[tag] for tag in self.instances[index]['tags']]
        para_ids = []
        for i in range(len(texts)):
            para_ids.append(self.para2id[PMID + "-" + str(i)] + 1)

        return torch.tensor(para_ids), torch.tensor(tags)


    def __len__(self):
        return len(self.instances)

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
    para_ids = [x[0] for x in batch]
    tags =  [x[1] for x in batch]
    length = [len(x[0]) for x in batch]
    para_ids = pad_sequence(para_ids, batch_first = True, padding_value = 0)
    tags = pad_sequence(tags, batch_first = True, padding_value = -1)
    return para_ids, tags, length


if __name__ == "__main__":
    data_dir = "../../../datasets/task_1_patient_finding"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    dataset = PatientFindingDataset(data_dir, "human")
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = MyCollateFn)
    import ipdb; ipdb.set_trace()