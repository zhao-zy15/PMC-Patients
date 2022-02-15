import torch
import numpy as np


tag2id = {"B": 1, "I": 2, "O": 0}

def find_all_ent(tags):
    ents = []
    start = -1
    for i in range(len(tags)):
        if tags[i] == 1:
            if start >= 0:
                ents.append((start, i - 1))
                start = -1
            start = i
        if tags[i] == 0:
            if start >= 0:
                ents.append((start, i - 1))
                start = -1
    if start >= 0:
        ents.append((start, len(tags)))
    return ents


def metric(golden_label, pred_label):
    assert len(pred_label) == len(golden_label)
    total_token = len(pred_label)
    right_token = sum([pred_label[i] == golden_label[i] for i in range(len(golden_label))])
    pred_entities = find_all_ent(pred_label)
    golden_entities = find_all_ent(golden_label)
    total_ent = len(golden_entities)
    pred_ent = len(pred_entities)
    right_ent = len(set(golden_entities) & set(pred_entities))
    return total_token, right_token, total_ent, pred_ent, right_ent


def batch_metric(golden_label, pred_label, length):
    total = np.array([0, 0, 0, 0, 0])
    for i in range(len(length)):
        pred_i = pred_label[i]
        golden_i = golden_label[i][:length[i]].detach().cpu().tolist()
        total += np.array(metric(golden_i, pred_i))
    return total


if __name__ == '__main__':
    golden_list = ["O", "O", "B", "I", "I", "B", "O", "O", "O", "O", "B", "I"]
    label_list =  ["I", "O", "B", "I", "B", "B", "O", "I", "I", "I", "B", "I"]
    
    print(find_all_ent([tag2id[x] for x in golden_list]))
    print(find_all_ent([tag2id[x] for x in label_list]))
    golden_list = [tag2id[x] for x in golden_list]
    label_list = [tag2id[x] for x in label_list]
    print(metric(golden_list, label_list))