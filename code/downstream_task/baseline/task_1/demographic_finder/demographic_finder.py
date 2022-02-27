import json
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("../../../../../PMC-Patients_collection/code/PMC-Patients_collection")
from filters import demo_filter
sys.path.append("..")
from utils import metric, para_metric

# Load data.
#PNR_human = json.load(open("../../../../../datasets/task_1_patient_note_recognition/PNR_human.json", "r"))
human_annotation = json.load(open("../../../../../datasets/task_1_patient_note_recognition/PNR_human_annotations.json", "r"))

tag2id = {"B": 1, "I": 2, "O": 3}
note_level = np.array([0,0,0,0,0])
para_level = np.array([0.,0.,0.])
for article in tqdm(human_annotation):
    true_tags = [tag2id[x] for x in article['tags']]
    pred_tags = []
    for para in article['texts']:
        # If has demographics, tag B, otherwise tag O.
        if demo_filter(para):
            pred_tags.append(tag2id["B"])
        else:
            pred_tags.append(tag2id["O"])
    note_level += np.array(metric(true_tags, pred_tags, tag2id))
    para_level += np.array(para_metric(true_tags, pred_tags, tag2id))

# Calculate metrics.
print("=====Note level=======")
precision = note_level[4] / note_level[3]
recall = note_level[4] / note_level[2]
f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1*100, precision*100, recall*100))


print("=====Paragraph level=======")
precision, recall, f1 = para_level / len(human_annotation)
print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1*100, precision*100, recall*100))