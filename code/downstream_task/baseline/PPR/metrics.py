import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score


def getRR(golden_list, results):
    for i in range(len(results)):
        if results[i] in golden_list:
            return 1 / (i + 1)
    return 0


def getPrecision(golden_list, results):
    acc_count = 0
    for result in results:
        if result in golden_list:
            acc_count += 1
    return acc_count / len(results)


def getRecall(golden_list, results):
    recall = 0
    for id in golden_list:
        if id in results:
            recall += 1
    return recall / len(golden_list)


def getNDCG(golden_list, results, scores):
    y_true = np.asarray([[1 if idx in golden_list else 0 for idx in results]])
    y_score = np.asarray([scores])
    return ndcg_score(y_true, y_score)


def getAP(golden_list, results, scores):
    y_true = np.asarray([1 if idx in golden_list else 0 for idx in results])
    if sum(y_true) == 0:
        return 0
    y_score = np.asarray(scores)
    return average_precision_score(y_true, y_score)


if __name__ == '__main__':
    golden_list = ['1', '2', '3']
    results = ['1', '2', '4']
    scores = [10, 9, 20]
    print(getNDCG(golden_list, results, scores))
    print(getAP(golden_list, results, scores))

    golden_list = ['1', '2', '3']
    results = ['11', '22', '4']
    scores = [10, 9, 20]
    print(getNDCG(golden_list, results, scores))
    print(getAP(golden_list, results, scores))
    
