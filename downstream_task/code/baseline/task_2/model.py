import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from dataloader import PatientSimilarityDataset, MyCollateFn
from torch.utils.data import DataLoader


class monoBERT(nn.Module):
    def __init__(self, model_name_or_path, mse):
        super(monoBERT, self).__init__()
        self.mse = mse
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        if self.mse:
            self.linear = nn.Sequential(
                nn.Linear(768, 1),
                nn.Tanh()
            )
        else:
            self.linear = nn.Linear(768, 3)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled = self.bert(input_ids, attention_mask, token_type_ids)[1]
        score = self.linear(pooled)
        if self.mse:
            score = score.squeeze()
        return pooled, score


if __name__ == "__main__":
    data_dir = "../../../datasets/task_2_patient_similarity"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 512)
    dataset = PatientSimilarityDataset(data_dir, "train", tokenizer)
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = MyCollateFn)
    
    it = iter(dataloader)
    input_ids, attention_mask, token_type_ids, label = it.next()
    model = monoBERT(model_name_or_path, False)
    import ipdb; ipdb.set_trace()
    model(input_ids, attention_mask, token_type_ids)