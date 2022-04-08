import json
import numpy as np 


human = json.load(open("PAR_PPR_anno.json", "r"))
label = json.load(open("PAR_human_eval_top_10.json", "r"))

total = [0, 0]
confusion = [[0, 0], [0, 0]]
MRR = []
P = []
P1 = []
for ins in label:
    patient_id = ins['query']['human_patient_uid']
    temp_mrr = 0
    temp_p = [0] * 5
    for i in range(5):
        candidate = ins['candidates'][i]
        PMID = candidate['PMID']
        ground_truth = human[patient_id][PMID]
        if ground_truth == '0':
            total[0] += 1
            if candidate['label'] == 0:
                confusion[0][0] += 1
            else:
                #import ipdb; ipdb.set_trace()
                confusion[0][1] += 1
        else:
            temp_p[i] = 1
            if temp_mrr == 0:
                temp_mrr = 1 / (i + 1)
            total[1] += 1
            if candidate['label'] == 1:
                confusion[1][1] += 1
            else:
                confusion[1][0] += 1
    MRR.append(temp_mrr)
    P.append(sum(temp_p) / 5)
    P1.append(temp_p[0])

print(total)
print(confusion)
print(np.mean(MRR))
print(np.mean(P))
print(np.mean(P1))
import ipdb; ipdb.set_trace()
