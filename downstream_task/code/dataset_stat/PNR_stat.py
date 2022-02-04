import json
from tqdm import tqdm,trange

train = json.load(open("../../datasets/task_1_patient_note_recognition/PNR_train.json", "r"))
dev = json.load(open("../../datasets/task_1_patient_note_recognition/PNR_dev.json", "r"))
test = json.load(open("../../datasets/task_1_patient_note_recognition/PNR_test.json", "r"))


article_count = 0
patient_count = 0
tag_count = {"B": 0, "I": 0, "O": 0, "E": 0, "S": 0}
total_tag_count = {"B": 0, "I": 0, "O": 0, "E": 0, "S": 0}
for patient in tqdm(train):
    article_count += 1
    for tag in patient['tags']:
        tag_count[tag] += 1
        total_tag_count[tag] += 1
        if tag == "B" or tag == "S":
            patient_count += 1
        
print("Article count: ", article_count)
print("Patient note count: ", patient_count)
total_tags = sum(tag_count.values())
print("Tags distribution: ", ["{} {:.2f}".format(k, v*100/total_tags) for k, v in tag_count.items()])

article_count = 0
patient_count = 0
tag_count = {"B": 0, "I": 0, "O": 0, "E": 0, "S": 0}
for patient in tqdm(dev):
    article_count += 1
    for tag in patient['tags']:
        tag_count[tag] += 1
        total_tag_count[tag] += 1
        if tag == "B" or tag == "S":
            patient_count += 1
        
print("Article count: ", article_count)
print("Patient note count: ", patient_count)
total_tags = sum(tag_count.values())
print("Tags distribution: ", ["{} {:.2f}".format(k, v*100/total_tags) for k, v in tag_count.items()])

article_count = 0
patient_count = 0
tag_count = {"B": 0, "I": 0, "O": 0, "E": 0, "S": 0}

for i in trange(len(test)):
    patient = test[i]
    article_count += 1
    for tag in patient['tags']:
        tag_count[tag] += 1
        total_tag_count[tag] += 1
        if tag == "B" or tag == "S":
            patient_count += 1

print("Article count: ", article_count)
print("Patient note count: ", patient_count)
total_tags = sum(tag_count.values())
print("Tags distribution: ", ["{} {:.2f}".format(k, v*100/total_tags) for k, v in tag_count.items()])

human_set = json.load(open("../../datasets/task_1_patient_note_recognition/PNR_human.json", "r"))
article_count = 0
patient_count = 0
tag_count = {"B": 0, "I": 0, "O": 0, "E": 0, "S": 0}
for patient in human_set:
    article_count += 1
    for tag in patient['tags']:
        tag_count[tag] += 1
        if tag == "B" or tag == "S":
            patient_count += 1

print("Article count: ", article_count)
print("Patient note count: ", patient_count)
total_tags = sum(tag_count.values())
print("Tags distribution: ", ["{} {:.2f}".format(k, v*100/total_tags) for k, v in tag_count.items()])


total_tags = sum(total_tag_count.values())
print("Total tags distribution: ", ["{} {:.2f}".format(k, v*100/total_tags) for k, v in total_tag_count.items()])
