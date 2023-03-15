import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib import cm
from matplotlib.font_manager import fontManager
fontManager.addfont("../../../../Gill_Sans_MT.ttf")
plt.rcParams['font.sans-serif'] = ['Gill Sans MT']


human = json.load(open("../../../datasets/task_3_patient2patient_retrieval/PPR_PAR_human_annotations.json", "r"))

PPR = [0] * 5
PAR = [0] * 5

for patient in human:
    for ins in human[patient]:
        value = human[patient][ins]
        for char in value:
            if "-" in ins:
                PPR[int(char)] += 1
            else:
                PAR[int(char)] += 1

PPR = np.array(PPR) / len(human) * 20
PAR = np.array(PAR) / len(human) * 20

norm = plt.Normalize(20, 80)

plt.rcParams['figure.figsize'] = (8.5,1.75)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
color = "steelblue"

fig = plt.figure()
ax1 = plt.axes([.05, .25, .4, .7])
ax2 = plt.axes([.56, .25, .4, .7])

ax1.bar([0,1,2,3,4], PAR, 0.3, color = color)
ax1.set_xticks([0,1,2,3,4])
ax1.set_xticklabels(["Irrelevant", "Diagnosis", "Test", "Treatment", "Others"], fontsize = 10)
ax1.set_xlabel("Patient-article relevance", fontsize = 12)
ax1.text(-0.75, 66, "%")

ax2.bar([0,1,2,3,4], PPR, 0.3, color = color)
ax2.set_xlabel("Patient-patient similarity", fontsize = 12)
ax2.set_xticks([0,1,2,3,4])
ax2.set_xticklabels(["Dissimilar", "Features", "Outcomes", "Exposure", "Others"], fontsize = 10)
ax2.text(-0.75, 75, "%")


plt.savefig("../../../figures/human_relations.pdf")
