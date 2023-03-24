import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from dataloader import PAR_BiEncoder_Dataset, MyCollateFn
from torch.utils.data import DataLoader


class BiEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super(BiEncoder, self).__init__()
        self.encoder_1 = AutoModel.from_pretrained(model_name_or_path)
        self.encoder_2 = AutoModel.from_pretrained(model_name_or_path)
        
        
    def forward(self, input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2, token_type_ids_2):
        q_vectors = self.encoder_1(input_ids = input_ids_1, attention_mask = attention_mask_1, token_type_ids = token_type_ids_1)[1]
        ctx_vectors = self.encoder_2(input_ids = input_ids_2, attention_mask = attention_mask_2, token_type_ids = token_type_ids_2)[1]
        scores = torch.matmul(q_vectors, ctx_vectors.transpose(1, 0))
        return scores


if __name__ == "__main__":
    data_dir = "../../../../../datasets/patient2article_retrieval"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = PPR_BiEncoder_Dataset(data_dir, "dev", tokenizer)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = MyCollateFn)
    
    it = iter(dataloader)
    model = BiEncoder(model_name_or_path)
    import ipdb; ipdb.set_trace()
    scores = model(*(it.next()))