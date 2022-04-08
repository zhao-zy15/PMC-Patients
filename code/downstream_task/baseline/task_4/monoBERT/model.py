import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F


class monoBERT(nn.Module):
    def __init__(self, model_name_or_path):
        super(monoBERT, self).__init__()
        self.model_name_or_path = model_name_or_path.lower()
        if "longformer" in self.model_name_or_path:
            self.bert_t = AutoModelForMaskedLM.from_pretrained(model_name_or_path).longformer
            self.bert_a = AutoModelForMaskedLM.from_pretrained(model_name_or_path).longformer
        else:
            self.bert_t = AutoModel.from_pretrained(model_name_or_path)
            self.bert_a = AutoModel.from_pretrained(model_name_or_path)
        self.linear = nn.Sequential(
            nn.Linear(768 * 2, 1),
            nn.ReLU()
        )
        
        
    def forward(self, t_input_ids, t_attention_mask, t_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids):
        if "longformer" in self.model_name_or_path:
            global_attention_mask = torch.zeros(t_input_ids.shape, device = t_input_ids.device()).long()
            global_attention_mask[:, 0] = 1
            t_pooled = self.bert_t(t_input_ids, t_attention_mask, global_attention_mask, output_hidden_states = True).hidden_states
            a_pooled = self.bert_a(a_input_ids, a_attention_mask, global_attention_mask, output_hidden_states = True).hidden_states
        else:
            t_pooled = self.bert_t(t_input_ids, t_attention_mask, t_token_type_ids)[1]
            a_pooled = self.bert_a(a_input_ids, a_attention_mask, a_token_type_ids)[1]
        pooled = torch.cat((t_pooled, a_pooled), dim = 1)
        score = self.linear(pooled)
        score = score.squeeze(1)
        return pooled, score


class MarginMSE(nn.Module):
    def __init__(self, p = 2, margin = 0.1, pos_ratio = 1):
        super(MarginMSE, self).__init__()
        self.p = p
        self.margin = margin
        self.pos_ratio = pos_ratio

    def forward(self, prob, label):
        pos_id = label == 1
        neg_id = label != 1
        pos_loss = torch.pow(1 - prob[pos_id], self.p)
        pos_loss[prob[pos_id] >= 1 - self.margin] = 0
        neg_loss = torch.pow(prob[neg_id], self.p)
        neg_loss[prob[neg_id] <= self.margin] = 0
        return (sum(pos_loss) * self.pos_ratio + sum(neg_loss)) / label.shape[0]


if __name__ == '__main__':
    x = torch.tensor([0, 0.05, 0.2, 0.8, 0.95, 1.1], requires_grad = True)
    y = torch.tensor([0, 0, 1, 0, 1, 1])
    lossFn = MarginMSE()
    loss = lossFn(x, y)
    loss.backward()
    import ipdb; ipdb.set_trace()
