import json
from tqdm import tqdm


train = json.load(open("../../../datasets/task_2_patient2patient_similarity/PPS_train.json", "r"))
ins_count = 0
sim_1_count = 0
sim_2_count = 0
neg_count = 0
ez_neg = 0
for ins in tqdm(train):
    ins_count += 1
    if ins[2] == 0:
        neg_count += 1
    if ins[2] == 1:
        sim_1_count += 1
    if ins[2] == 2:
        sim_2_count += 1
    if ins[2] == -1:
        ez_neg += 1
        # Convert -1 label to 0.
        ins[2] = 0

json.dump(train, open("../../../datasets/task_2_patient2patient_similarity/PPS_train.json", "w"), indent = 4)

print("Instance count: ", ins_count)
print("Similarity 1 count: ", sim_1_count)
print("Similarity 2 count: ", sim_2_count)
print("Negative sample count: ", neg_count)
print("Easy negative:", ez_neg)

dev = json.load(open("../../../datasets/task_2_patient2patient_similarity/PPS_dev.json", "r"))
ins_count = 0
sim_1_count = 0
sim_2_count = 0
neg_count = 0
ez_neg = 0
for ins in tqdm(dev):
    ins_count += 1
    if ins[2] == 0:
        neg_count += 1
    if ins[2] == 1:
        sim_1_count += 1
    if ins[2] == 2:
        sim_2_count += 1
    if ins[2] == -1:
        ez_neg += 1
        ins[2] = 0

print("Instance count: ", ins_count)
print("Similarity 1 count: ", sim_1_count)
print("Similarity 2 count: ", sim_2_count)
print("Negative sample count: ", neg_count)
print("Easy negative:", ez_neg)

json.dump(dev, open("../../../datasets/task_2_patient2patient_similarity/PPS_dev.json", "w"), indent = 4)

test = json.load(open("../../../datasets/task_2_patient2patient_similarity/PPS_test.json", "r"))
ins_count = 0
sim_1_count = 0
sim_2_count = 0
neg_count = 0
ez_neg = 0
for ins in tqdm(test):
    ins_count += 1
    if ins[2] == 0:
        neg_count += 1
    if ins[2] == 1:
        sim_1_count += 1
    if ins[2] == 2:
        sim_2_count += 1
    if ins[2] == -1:
        ez_neg += 1
        ins[2] = 0
        
print("Instance count: ", ins_count)
print("Similarity 1 count: ", sim_1_count)
print("Similarity 2 count: ", sim_2_count)
print("Negative sample count: ", neg_count)
print("Easy negative:", ez_neg)

json.dump(test, open("../../../datasets/task_2_patient2patient_similarity/PPS_test.json", "w"), indent = 4)
