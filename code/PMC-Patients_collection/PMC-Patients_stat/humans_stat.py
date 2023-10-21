import json


data = json.load(open("../../../datasets/PMC-Patients-Humans.json", "r"))
PMID2Mesh = json.load(open("../../../meta_data/PMID2Mesh.json", "r"))

ages = ["Infant, Newborn", "Infant", "Child, Preschool", "Child", "Adolescent", "Young Adult", "Adult", "Middle Aged", "Aged", "Aged, 80 and over"]
age_counts = {age: 0 for age in ages}
total = 0

for patient in data:
    mesh = PMID2Mesh[patient['PMID']]
    if len(set(mesh) & set(ages)) != 1:
        continue
    total += 1
    for age in ages:
        if age in mesh:
            age_counts[age] += 1

print(total)
print(age_counts)


def get_age(age):
    result = 0
    for age_x in age:
        num, unit = age_x
        if unit == "year":
            result += num
        if unit == "month":
            result += num / 12
        if unit == "week":
            result += num / 52
        if unit == "day":
            result += num / 365
    return result


age_ac = 0
age_split = [1 / 12, 23 / 12, 5, 12, 18, 24, 44, 64, 79]
for patient in data:
    mesh = PMID2Mesh[patient['PMID']]
    if len(set(mesh) & set(ages)) != 1:
        continue
    age = get_age(patient['age'])
    if age <= 28 / 365 and ages[0] in mesh:
        age_ac += 1
    for i in range(8):
        if age >= age_split[i] and age <= age_split[i+1] and ages[i+1] in mesh:
            age_ac += 1
    if age >= age_split[-1] and ages[9] in mesh:
        age_ac += 1

print(age_ac, total, age_ac / total)



total = 0
male = 0
female = 0

for patient in data:
    mesh = PMID2Mesh[patient['PMID']]
    if int("Male" in mesh) + int("Female" in mesh) != 1:
        continue
    total += 1
    if "Male" in mesh:
        male += 1
    else:
        female += 1

print(total)
print(male, female)

gender_ac = 0
for patient in data:
    mesh = PMID2Mesh[patient['PMID']]
    if int("Male" in mesh) + int("Female" in mesh) != 1:
        continue
    if patient['gender'][0] == "M" and "Male" in mesh:
        gender_ac += 1
    if patient['gender'][0] == "F" and "Female" in mesh:
        gender_ac += 1

print(gender_ac, total, gender_ac / total)

