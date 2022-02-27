import torch
import numpy as np


"""
    Return the start of end index of valid entity spans in given tags.
"""
def find_all_ent(tags, tag2id):
    ents = []
    start = -1
    for i in range(len(tags)):
        if tags[i] == tag2id["B"]:
            if start >= 0:
                ents.append((start, i - 1))
                start = -1
            start = i
        if tags[i] == tag2id["O"]:
            if start >= 0:
                ents.append((start, i - 1))
                start = -1
    if start >= 0:
        ents.append((start, len(tags)))
    return ents


def getPrecision(golden_span, pred_span):
    golden_set = set(range(golden_span[0], golden_span[1] + 1))
    pred_set = set(range(pred_span[0], pred_span[1] + 1))
    return len(golden_set & pred_set) / len(pred_set)


def getRecall(golden_span, pred_span):
    golden_set = set(range(golden_span[0], golden_span[1] + 1))
    pred_set = set(range(pred_span[0], pred_span[1] + 1))
    return len(golden_set & pred_set) / len(golden_set)


"""
    Note level counts.
"""
def metric(golden_label, pred_label, tag2id):
    assert len(pred_label) == len(golden_label)
    total_token = len(pred_label)
    right_token = sum([pred_label[i] == golden_label[i] for i in range(len(golden_label))])
    pred_entities = find_all_ent(pred_label, tag2id)
    golden_entities = find_all_ent(golden_label, tag2id)
    total_ent = len(golden_entities)
    pred_ent = len(pred_entities)
    right_ent = len(set(golden_entities) & set(pred_entities))

    return total_token, right_token, total_ent, pred_ent, right_ent


"""
    Paragraph level metrics.
"""
def para_metric(golden_label, pred_label, tag2id):
    assert len(pred_label) == len(golden_label)
    pred_entities = find_all_ent(pred_label, tag2id)
    golden_entities = find_all_ent(golden_label, tag2id)
    precision = []
    for pred in pred_entities:
        precision.append(0)
        for true in golden_entities:
            P = getPrecision(true, pred)
            if P > precision[-1]:
                precision[-1] = P
    recall = []
    for true in golden_entities:
        recall.append(0)
        for pred in pred_entities:
            R = getRecall(true, pred)
            if R > recall[-1]:
                recall[-1] = R
    precision = np.mean(precision) if len(precision) > 0 else 1
    recall = np.mean(recall) if len(recall) > 0 else 1
    F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    
    return precision, recall, F1


"""
    For batch input.
"""
def batch_metric(golden_label, pred_label, length, tag2id):
    total = np.array([0, 0, 0, 0, 0])
    for i in range(len(length)):
        pred_i = pred_label[i]
        golden_i = golden_label[i][:length[i]].detach().cpu().tolist()
        total += np.array(metric(golden_i, pred_i, tag2id))
    return total


"""
    For batch input.
"""
def batch_para_metric(golden_label, pred_label, length, tag2id):
    metrics = np.array([0.,0.,0.])
    for i in range(len(length)):
        pred_i = pred_label[i]
        golden_i = golden_label[i][:length[i]].detach().cpu().tolist()
        metrics += np.array(para_metric(golden_i, pred_i, tag2id))
    return metrics


if __name__ == '__main__':
    golden_list = ["O", "O", "B", "I", "I", "B", "O", "O", "O", "O", "B", "I"]
    label_list =  ["I", "O", "B", "I", "B", "B", "O", "I", "I", "I", "B", "I"]
    tag2id = {"B": 1, "I": 2, "O": 3}
    print(find_all_ent([tag2id[x] for x in golden_list], tag2id))
    print(find_all_ent([tag2id[x] for x in label_list], tag2id))
    golden_list = [tag2id[x] for x in golden_list]
    label_list = [tag2id[x] for x in label_list]
    print(metric(golden_list, label_list, tag2id))
    print(para_metric(golden_list, label_list, tag2id))

    golden_list = ["O", "O", "B", "I", "I", "B"]
    label_list =  ["O", "O", "B", "I", "B", "O"]
    golden_list = [tag2id[x] for x in golden_list]
    label_list = [tag2id[x] for x in label_list]
    print(metric(golden_list, label_list, tag2id))
    print(para_metric(golden_list, label_list, tag2id))