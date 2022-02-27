import torch
from torch import nn
import torch.nn.functional as F
from dataloader import PatientFindingDataset, MyCollateFn
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import argparse


def get_mask(length):
    max_len = int(max(length))
    mask = torch.Tensor()
    # length = length.numpy()
    for len_ in length:
        mask = torch.cat((mask, torch.Tensor([[1] * len_ + [0] * (max_len-len_)])), dim=0)
    return mask.transpose(0, 1)


def get_cut_points(length):
    cut_points = [0]
    cur = 0
    for l in length:
        cur += l
        cut_points.append(cur)
    return cut_points


class CRFLayer(nn.Module):
    def __init__(self, n_class, tag2id):
        super(CRFLayer, self).__init__()
        # transition[i][j] means transition probability from j to i
        self.n_class = n_class
        self.tag2id = tag2id
        self.transition = nn.Parameter(torch.randn(n_class, n_class))

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.transition)
        # initialize '<SOS>', '<EOS>' probability in log space
        self.transition.detach()[self.tag2id['<SOS>'], :] = -10000
        self.transition.detach()[:, self.tag2id['<EOS>']] = -10000

    def forward(self, feats, mask):
        """
        Arg:
        feats: (seq_len, batch_size, n_class)
        mask: (seq_len, batch_size)
        Return:
        scores: (batch_size, )
        """
        seq_len, batch_size, n_class = feats.size()
        # initialize alpha to zero in log space
        alpha = feats.new_full((batch_size, n_class), fill_value=-10000)
        # alpha in '<SOS>' is 1
        alpha[:, self.tag2id['<SOS>']] = 0
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # emit_score is the same regardless of current_tag, so we broadcast along current_tag
            emit_score = feat.unsqueeze(-1) # (batch_size, n_class, 1)
            # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
            transition_score = self.transition.unsqueeze(0) # (1, n_class, n_class)
            # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
            alpha_score = alpha.unsqueeze(1) # (batch_size, 1, n_class)
            alpha_score = alpha_score + transition_score + emit_score
            # torch.logsumexp along current_tag dimension to get next_tag alpha
            mask_t = mask[t].unsqueeze(-1)
            alpha = torch.logsumexp(alpha_score, -1) * mask_t + alpha * (1 - mask_t) # (batch_size, n_class)
        # arrive at '<EOS>'
        alpha = alpha + self.transition[self.tag2id['<EOS>']].unsqueeze(0)

        return torch.logsumexp(alpha, -1) # (batch_size, )

    def score_sentence(self, feats, tags, mask):
        """
        Arg:
        feats: (seq_len, batch_size, n_class)
        tags: (seq_len, batch_size)
        mask: (seq_len, batch_size)
        Return:
        scores: (batch_size, )
        """
        seq_len, batch_size, n_class = feats.size()
        scores = feats.new_zeros(batch_size)
        tags = torch.cat([tags.new_full((1, batch_size), fill_value=self.tag2id['<SOS>']), tags], 0) # (seq_len + 1, batch_size)
        for t, feat in enumerate(feats):
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])
            transition_score = torch.stack([self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])
            scores += (emit_score + transition_score) * mask[t]
        transition_to_end = torch.stack([self.transition[self.tag2id['<EOS>'], tag[mask[:, b].sum().long()]] for b, tag in enumerate(tags.transpose(0, 1))])
        scores += transition_to_end
        return scores

    def viterbi_decode(self, feats, mask):
        """
        :param feats: (seq_len, batch_size, n_class)
        :param mask: (seq_len, batch_size)
        :return best_path: (seq_len, batch_size)
        """
        seq_len, batch_size, n_class = feats.size()
        # initialize scores in log space
        scores = feats.new_full((batch_size, n_class), fill_value=-10000)
        scores[:, self.tag2id['<SOS>']] = 0
        pointers = []
        # forward
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, n_class, n_class)
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, n_class), (batch_size, n_class)
            scores_t += feat
            pointers.append(pointer)
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            scores = scores_t * mask_t + scores * (1 - mask_t)
        pointers = torch.stack(pointers, 0) # (seq_len, batch_size, n_class)
        scores += self.transition[self.tag2id['<EOS>']].unsqueeze(0)
        best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )
        # backtracking
        best_path = best_tag.unsqueeze(-1).tolist() # list shape (batch_size, 1)
        for i in range(batch_size):
            best_tag_i = best_tag[i]
            seq_len_i = int(mask[:, i].sum())
            for ptr_t in reversed(pointers[:seq_len_i, i]):
                # ptr_t shape (n_class, )
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # pop first tag
            best_path[i].pop()
            # reverse order
            best_path[i].reverse()
        return best_path


class CNN_LSTM_CRF(nn.Module):
    def __init__(self, args, pretrained_embedding = None):
        super(CNN_LSTM_CRF, self).__init__()
        self.embed_dim = 768
        self.vocab_size = args.vocab_size
        self.kernel_num = args.kernel_num
        self.kernel_size = args.kernel_size
        self.hidden_dim = 64
        self.tag2id = {"<SOS>": 0, "B": 1, "I": 2, "O": 3, "<EOS>": 4}
        self.n_class = len(self.tag2id)
        if pretrained_embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx = 0)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze = args.freeze, padding_idx = 0)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (size, self.embed_dim)) for size in self.kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        self.rnn = nn.LSTM(input_size=self.kernel_num * len(self.kernel_size), hidden_size=self.hidden_dim, \
            num_layers=2, bidirectional=True, dropout = args.dropout)
        self.linear = nn.Linear(2*self.hidden_dim, self.n_class)
        self.crf = CRFLayer(self.n_class, self.tag2id)

    def get_lstm_features(self, input_ids, para_len, mask):
        """
        :param seq: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        
        embeded_input = self.embedding(input_ids) # [sum(para_len), sent_len, embed_dim(768)]
        embeded_input = embeded_input.unsqueeze(1) #[sum(para_len), 1, sent_len, embed_dim(768)]
        conved = [F.relu(conv(embeded_input)).squeeze(3) for conv in self.convs] # [len(kernel_size), sum(para_len), kernel_num, sent_len]
        pooled = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in conved] # [len(kernel_size), sum(para_len), kernel_num]
        para_embed = torch.cat(pooled, 1) #[sum(para_len), len(kernel_size)*kernel_num]
        #para_embed = self.dropout(para_embed)
        cut_points = get_cut_points(para_len)
        batch_paras = [para_embed[cut_points[i]:cut_points[i+1], :] for i in range(len(cut_points) - 1)]
        batch_paras = pad_sequence(batch_paras, batch_first = True).transpose(0, 1) # [max_para_len, batch, len(kernel_size)*kernel_num]
        packed_embedded_input = pack_padded_sequence(input=batch_paras, lengths=para_len, batch_first=False, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_embedded_input)
        output, _ = pad_packed_sequence(packed_output)
        lstm_output = output * mask.unsqueeze(-1)
        lstm_features = self.linear(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, n_class)
        return lstm_features


    def neg_log_likelihood(self, seq, tags, mask, para_len):
        """
        :param seq: (seq_len, batch_size)
        :param tags: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        lstm_features = self.get_lstm_features(seq, para_len, mask)
        forward_score = self.crf(lstm_features, mask)
        gold_score = self.crf.score_sentence(lstm_features, tags, mask)
        loss = (forward_score - gold_score).sum()

        return loss

    def predict(self, seq, mask, para_len):
        """
        :param seq: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        lstm_features = self.get_lstm_features(seq, para_len, mask)
        best_paths = self.crf.viterbi_decode(lstm_features, mask)

        return best_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel_num",
        default = 3,
        type = int,
        help = "Number of kernels."
    )
    parser.add_argument(
        "--kernel_size",
        default = [2,3,4],
        type = list,
        help = "Sizes of kernels."
    )
    parser.add_argument(
        "--dropout",
        default = 0.3,
        type = float,
        help = "Dropout."
    )
    args = parser.parse_args()

    data_dir = "../../../../../datasets/task_1_patient_note_recognition"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 256)
    args.vocab_size = tokenizer.vocab_size
    dataset = PatientFindingDataset(tokenizer, 256, data_dir, "test")
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = MyCollateFn)
    
    device = "cuda:0"
    it = iter(dataloader)
    input_ids, tags, length = it.next()
    input_ids = input_ids.to(device)
    tags = tags.to(device).transpose(0, 1)
    model = CNN_LSTM_CRF(args)
    model.to(device)
    
    mask = get_mask(length).to(device)
    loss = model.neg_log_likelihood(input_ids, tags, mask, length)
    pred = model.predict(input_ids, mask, length)
    
    import ipdb; ipdb.set_trace()
    