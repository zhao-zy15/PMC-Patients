import torch


tag2id = {"<SOS>": 0, "B": 1, "I": 2, "O": 3, "E": 4, "S": 5, "<EOS>": 6}


def find_all_ent(tags):
    ents = []
    start = -1
    for i in range(len(tags)):
        if tags[i] == tag2id["S"]:
            ents.append((i, i))
            start = -1
        if tags[i] == tag2id["O"]:
            start = -1
        if tags[i] == tag2id["B"]:
            start = i
        if tags[i] == tag2id["E"] and start > 0:
            ents.append((start, i))
            start = -1
    return ents


def metric(tags, pred, length):
    total_token = 0
    right_token = 0
    total_ent = 0
    pred_ent = 0
    right_ent = 0
    for i in range(tags.shape[0]):
        pred_label = torch.stack(pred[i]).detach().cpu().numpy()
        golden_label = tags[i][:length[i]].detach().cpu().numpy()
        assert len(pred_label) == len(golden_label)
        total_token += len(pred_label)
        right_token += sum(pred_label == golden_label)
        pred_entities = find_all_ent(pred_label)
        golden_entities = find_all_ent(golden_label)
        total_ent += len(golden_entities)
        pred_ent += len(pred_entities)
        right_ent += len(set(golden_entities) & set(pred_entities))
    return total_token, right_token, total_ent, pred_ent, right_ent


if __name__ == '__main__':
    golden_list = ["O", "O", "B", "I", "E", "S", "O", "O", "O", "O", "O", "O"]
    label_list =  ["I", "O", "B", "E", "B", "S", "O", "E", "I", "E", "B", "I"]
    print(find_all_ent([tag2id[x] for x in golden_list]))
    print(find_all_ent([tag2id[x] for x in label_list]))
    label_list = [[torch.tensor(tag2id[x]) for x in label_list]]
    golden_list = torch.tensor([tag2id[x] for x in golden_list]).unsqueeze(0)
    print(metric(golden_list, label_list, [12]))