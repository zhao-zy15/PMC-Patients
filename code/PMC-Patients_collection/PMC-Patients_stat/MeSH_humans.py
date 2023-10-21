import json


PMID2Mesh = json.load(open("../../../meta_data/PMID2Mesh.json", "r"))
PMIDs_humans = set([item[0] for item in PMID2Mesh.items() if "Humans" in item[1]])
print(len(PMIDs_humans))

data = json.load(open("../../../datasets/PMC-Patients.json", "r"))
new_data = [patient for patient in data if patient['PMID'] in PMIDs_humans]
print(len(new_data))

# no_mesh_patients = [patient for patient in data if patient['PMID'] not in PMID2Mesh]
# print(len(no_mesh_patients))
# import ipdb; ipdb.set_trace()
json.dump(new_data, open("../../../datasets/PMC-Patients-Humans.json", "w"), indent = 4)
