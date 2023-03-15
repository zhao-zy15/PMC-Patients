import numpy as np


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


def getDCG(scores):
    return np.sum(
        np.divide(scores, np.log2(np.arange(len(scores), dtype=np.float32) + 2)),
        dtype=np.float32)

    
def getNDCG(rel_1, rel_2, results):
    result_scores = []
    for result in results:
        if result in rel_1:
            result_scores.append(1)
        elif result in rel_2:
            result_scores.append(2)
        else:
            result_scores.append(0)
    DCG = getDCG(result_scores)
    idea_scores = [2] * len(rel_2) + [1] * len(rel_1)
    if len(idea_scores) < len(results):
        idea_scores += [0] * (len(results) - len(idea_scores))
    else:
        idea_scores = idea_scores[:len(results)]
    assert len(idea_scores) == len(result_scores)
    IDCG = getDCG(idea_scores)
    return DCG / IDCG



if __name__ == '__main__':
    scores = [3,2,3,0,1,2]
    print(getDCG(scores))
    print(getDCG([3,3,3,2,2,1]))
    print(getDCG(scores) / getDCG([3,3,3,2,2,1]))

