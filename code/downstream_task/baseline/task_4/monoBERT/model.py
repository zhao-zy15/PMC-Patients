import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class monoBERT(nn.Module):
    def __init__(self, model_name_or_path):
        super(monoBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.linear = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled = self.bert(input_ids, attention_mask, token_type_ids)[1]
        score = self.linear(pooled)
        score = score.squeeze(1)
        return pooled, score

