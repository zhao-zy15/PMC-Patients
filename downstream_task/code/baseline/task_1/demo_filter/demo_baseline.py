import json
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("../../../../../PMC-Patients_collection/code/PMC-Patients_collection")
from filters import demo_filter
from utils import metric

# Load data.
PNR_test = json.load(open("../../../../datasets/task_1_patient_note_recognition/PNR_test.json", "r"))

tag2id = {"B": 1, "I": 2, "O": 3}
#total_token, right_token, total_ent, pred_ent, right_ent
counts = np.array([0,0,0,0,0])
for article in tqdm(PNR_test):
    true_tags = [tag2id[x] for x in article['tags']]
    pred_tags = []
    for para in article['texts']:
        # If has demographics, tag B, otherwise tag O.
        if demo_filter(para):
            pred_tags.append(1)
        else:
            pred_tags.append(0)
    counts += np.array(metric(true_tags, pred_tags))

# Calculate metrics.
token_acc = counts[1] / counts[0]
precision = counts[4] / counts[3] if counts[3] != 0 else 0
recall = counts[4] / counts[2]
f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

print("Acc:{:.3f}, F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(token_acc*100, f1*100, precision*100, recall*100))