import json
import os
import numpy as np


meta_data_dir = "../../meta_data"

relevant_article = json.load(open(os.path.join(meta_data_dir, "patient2article_relevance.json"), "r"))

print("====== Article Relevance Graph =======")
print("Article count: ", len(relevant_article))
length = [len(item) for item in relevant_article.values()]
print("Relevance count: ", sum(length))
print("Average Relevance per article: ", np.mean(length))
print("Min Relevance per article: ", np.min(length))
print("Number of 1:", np.bincount(length)[1])
print("Max Relevance per article: ", np.max(length))

patient_similarity = json.load(open(os.path.join(meta_data_dir, "patient2patient_similarity.json"), "r"))
patient_similarity = {k: v[0] + v[1] for k,v in patient_similarity.items()}

print("====== Patient Similarity Graph =======")
print("Nodes count: ", len(patient_similarity))
length = [len(item) for item in patient_similarity.values()]
print("Edges count: ", sum(length))
print("Average Degree: ", np.mean(length))
print("Isolated nodes count:", np.bincount(length)[0])

# Utilize union-find disjoint set to find subgraphs.
def find(node):
    root = node
    while father[root] != root:
        root = father[root]
    return root

def merge(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        father[root_x] = root_y

# Merge connected nodes into a set.
father = {x: x for x in patient_similarity.keys()}
for patient in patient_similarity.keys():
    for similar_patient in patient_similarity[patient]:
        merge(patient, similar_patient)

# Count node number in each disjoint set.
roots = [find(x) for x in patient_similarity.keys()]
roots_num = {}
for root in roots:
    if root in roots_num.keys():
        roots_num[root] += 1
    else:
        roots_num[root] = 1

print("Number of disjoint subgraphs: ", len(roots_num))
print("Nodes in Maximum clique: ", max(roots_num.values()))

