import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from dataloader import PatientFindingDataset, MyCollateFn
from torch.utils.data import DataLoader


class BERT_classify(nn.Module):
    def __init__(self, model_name_or_path):
        super(BERT_classify, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.linear = nn.Linear(768, 3)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled = self.bert(input_ids, attention_mask, token_type_ids)[1]
        score = self.linear(pooled)
        return pooled, score


if __name__ == "__main__":
    data_dir = "../../../../../datasets/task_1_patient_note_recognition"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 512)
    dataset = PatientFindingDataset(data_dir, tokenizer, "test", 512)
    batch_size = 24
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn = MyCollateFn)
    
    it = iter(dataloader)
    input_ids, attention_mask, token_type_ids, tags, index = it.next()
    model = BERT_classify(model_name_or_path)
    import ipdb; ipdb.set_trace()
    pooled, score = model(input_ids, attention_mask, token_type_ids)