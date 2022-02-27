import json
import re
import os
from tqdm import trange, tqdm
from word2number import w2n

# Usual age pattern such as "3 years old", "1-year- and 2-month-old"
age_pattern = r"(((?P<year>[0-9]+\.?[0-9]?([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]*((y((ear)|r)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<month>[0-9]+\.?[0-9]?([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]*((m(onth)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<week>[0-9]+\.?[0-9]?([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]*((w(eek)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<day>[0-9]+\.?[0-9]?([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]*((d(ay)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<hour>[0-9]+\.?[0-9]?([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]*((h(our)?)s?)))+"
age_pattern1 = re.compile(age_pattern + "[ /\._\-\‐](o(ld)?)[^a-z]")
# Age pattern using words, such as "nine years old", "thirteen years old"
word_age_pattern = r"(((?P<year>((twenty)|(thirty)|(forty)|(fifty)|(sixty)|(seventy)|(eighty)|(ninety))?[ /\._\-\‐]*[a-z]*([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]((y((ear)|r)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<month>((twenty)|(thirty)|(fifty)|(sixty)|(seventy)|(eighty)|(ninety))?[ /\._\-\‐]*[a-z]*([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]((m(onth)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<week>((twenty)|(thirty)|(forty)|(fifty)|(sixty)|(seventy)|(eighty)|(ninety))?[ /\._\-\‐]*[a-z]*([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]((w(eek)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<day>((twenty)|(thirty)|(forty)|(fifty)|(sixty)|(seventy)|(eighty)|(ninety))?[ /\._\-\‐]*[a-z]*([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]((d(ay)?)s?)([ /\._\-\‐]*and[ /\._\-\‐])?)|((?P<hour>((twenty)|(thirty)|(forty)|(fifty)|(sixty)|(seventy)|(eighty)|(ninety))?[ /\._\-\‐]*[a-z]*([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[ /\._\-\‐]((h(our)?)s?)))+"
word_age_pattern1 = re.compile(word_age_pattern + "[ /\._\-\‐](o(ld)?)[^a-z]")
# "Male aged 51 years"
age_pattern2 = re.compile(r"(^|[^a-z])((male)|((gentle)?(police)?man)|(boy)|(female)|(lady)|(girl)|(housewife)|((police)?woman)|(.*gravida)|(infant)|(baby)|(child)|(patient)),? aged " + age_pattern + r"[^a-z]")
# "A boy, aged 8"
age_pattern3 = re.compile(r"(^|[^a-z])((male)|((gentle)?(police)?man)|(boy)|(female)|(lady)|(girl)|(housewife)|((police)?woman)|(.*gravida)|(infant)|(baby)|(child)|(patient)),? aged (?P<year>[0-9]+\.?[0-9]?)[^a-z]")
# "Male aged forty six years"
word_age_pattern2 = re.compile(r"(^|[^a-z])((male)|((gentle)?(police)?man)|(boy)|(female)|(lady)|(girl)|(housewife)|((police)?woman)|(.*gravida)|(infant)|(baby)|(child)|(patient)),? aged " + word_age_pattern + r"[^a-z]")
# "Boy aged eight"
word_age_pattern3 = re.compile(r"(^|[^a-z])((male)|((gentle)?(police)?man)|(boy)|(female)|(lady)|(girl)|(housewife)|((police)?woman)|(.*gravida)|(infant)|(baby)|(child)|(patient)),? aged (?P<year>((twenty)|(thirty)|(forty)|(fifty)|(sixty)|(seventy)|(eighty)|(ninety))?[ /\._\-\‐]*[a-z]*([ /\._\-\‐]and[ /\._\-\‐](((a)|(one))[ /\._\-\‐])?half)?)[^a-z]")
# Detect words indicating male.
male_pattern = re.compile(r"(^|[^a-z])((he)|(male)|((gentle)?(police)?man)|(boy)|(prostat[a-z]*)|(mr))[^a-z]")
# Detect words indication female.
female_pattern = re.compile(r"(^|[^a-z])((she)|(female)|(lady)|(girl)|(housewife)|((police)?woman)|([a-z]*gravida)|(pregnan[a-z]*)|((g[0-9])|(p[0-9]))|(mentrua[a-z]*)|(uteri[a-z]*)|(mrs)|(ms))[^a-z]")
# Detect words indicating groups of males. (If group, filter)
males_pattern = re.compile(r"(^|[^a-z])((males)|((gentle)?men)|(boys))[^a-z]")
# Detect words indicating groups of females. (If group, filter)
females_pattern = re.compile(r"(^|[^a-z])((females)|(ladies)|(girls)|(women))[^a-z]")
# "Male in his (early/late) 70s"
age_pattern4 = re.compile(r"(^|[^a-z])((male)|((gentle)?(police)?man)|(boy)|(female)|(lady)|(girl)|(housewife)|((police)?woman)|(.*gravida)|(infant)|(baby)|(child)|(patient)),? in ((his)|(her)) (?P<time>((early)|(late)) )?(?P<year>[0-9]0s)[^a-z]")
# "Male in his (early/late) twenties"
word_age_pattern4 = re.compile(r"(^|[^a-z])((male)|((gentle)?(police)?man)|(boy)|(female)|(lady)|(girl)|(housewife)|((police)?woman)|(.*gravida)|(infant)|(baby)|(child)|(patient)),? in ((his)|(her)) (?P<time>((early)|(late)) )?(?P<year>((twenties)|(thirties)|(fourties)|(fifties)|(sixties)|(seventies)|(eighties)|(nineties)))[^a-z]")


"""
    Extract and return age of the patient.
    Input:
        text: patient note candidate
    Output:
        list of ages of different units, each is a tuple (number, unit).
        e.g. [[1.0, "year"], ["3.0", "month"]]
        Note that age is a float since there could be input "one and a half year" with output '[[1.5, "year"]]'
"""
def age_extract(text):
    results = []
    age = age_pattern1.search(text)
    if age:
        for unit in ['year', 'month', 'week', 'day', 'hour']:
            if age.group(unit):
                if "half" in age.group(unit):
                    temp = re.search(r"[0-9]+\.?[0-9]?", age.group(unit)).group()
                    results.append([float(temp) + 0.5, unit])
                else:
                    results.append([float(age.group(unit)), unit])
    word_age = word_age_pattern1.search(text.replace('fourty', 'forty').replace('ninty', 'ninety'))
    if word_age and len(results) == 0:
        for unit in ['year', 'month', 'week', 'day', 'hour']:
            if word_age.group(unit):
                try:
                    results.append([float(w2n.word_to_num(word_age.group(unit))), unit])
                except Exception as e:
                    continue
                if "half" in word_age.group(unit):
                    results[-1][0] += 0.5

    age = age_pattern2.search(text)
    if age and len(results) == 0:
        for unit in ['year', 'month', 'week', 'day', 'hour']:
            if age.group(unit):
                if "half" in age.group(unit):
                    temp = re.search(r"[0-9]+\.?[0-9]?", age.group(unit)).group()
                    results.append([float(temp) + 0.5, unit])
                else:
                    results.append([float(age.group(unit)), unit])
    word_age = word_age_pattern2.search(text.replace('fourty', 'forty').replace('ninty', 'ninety'))
    if word_age and len(results) == 0:
        for unit in ['year', 'month', 'week', 'day', 'hour']:
            if word_age.group(unit):
                try:
                    results.append([float(w2n.word_to_num(word_age.group(unit))), unit])
                except Exception as e:
                    continue
                if "half" in word_age.group(unit):
                    results[-1][0] += 0.5

    age = age_pattern3.search(text)
    if age and len(results) == 0:
        for unit in ['year']:
            if age.group(unit):
                if "half" in age.group(unit):
                    temp = re.search(r"[0-9]+\.?[0-9]?", age.group(unit)).group()
                    results.append([float(temp) + 0.5, unit])
                else:
                    results.append([float(age.group(unit)), unit])
    word_age = word_age_pattern3.search(text.replace('fourty', 'forty').replace('ninty', 'ninety'))
    if word_age and len(results) == 0:
        for unit in ['year']:
            if word_age.group(unit):
                try:
                    results.append([float(w2n.word_to_num(word_age.group(unit))), unit])
                except Exception as e:
                    continue
                if "half" in word_age.group(unit):
                    results[-1][0] += 0.5

    age = age_pattern4.search(text)
    if age and len(results) == 0:
        for unit in ['year']:
            if age.group(unit):
                results.append([float(age.group(unit)[:-1]), unit])
                if age.group('time'):
                    if 'early' in age.group('time'):
                        results[-1][0] += 2.5
                    else:
                        results[-1][0] += 7.5
                else:
                    results[-1][0] += 5
    word_age = word_age_pattern4.search(text.replace('fourties', 'forties').replace('ninties', 'nineties'))
    if word_age and len(results) == 0:
        for unit in ['year']:
            if word_age.group(unit):
                try:
                    results.append([float(w2n.word_to_num(word_age.group(unit).replace('ties', 'ty'))), unit])
                except Exception as e:
                    continue
                if word_age.group('time'):
                    if 'early' in word_age.group('time'):
                        results[-1][0] += 2.5
                    else:
                        results[-1][0] += 7.5
                else:
                    results[-1][0] += 5
    return results

"""
    Extract gender. Note when both male and female, or males / females are detected, the candidate is filtered.
    Input:
        patient note candidate
    Output:
        "M" or "F"
"""
def gender_extract(text):
    male_match = male_pattern.search(text)
    males_match = males_pattern.search(text)
    female_match = female_pattern.search(text)
    females_match = females_pattern.search(text)
    # If both male and female, or males / females are detected, filter the candidate
    if (male_match or males_match) and (female_match or females_match):
        male_span = male_match.span() if (not males_match) or (male_match and males_match and males_match.span()[0] > male_match.span()[0]) else males_match.span()
        female_span = female_match.span() if (not females_match) or (female_match and females_match and females_match.span()[0] > female_match.span()[0]) else females_match.span()
        if min(abs(female_span[0] - male_span[1]), abs(female_span[1] - male_span[0])) < 20:
            return None

    if male_pattern.search(text):
        return "M"
    else:
        if female_pattern.search(text):
            return "F"
        else:
            return None

"""
    Extract and return age and gender. 
    Input:
        Patient note candidate.
    Output:
        See functions above.
"""
def demo_filter(text):
    text = text.strip().lower()
    texts = text.split('. ')
    text = texts[0] + '. '
    for i in range(0, len(texts) - 1):
        if len(texts[i].strip().split()) < 10:
            text += texts[i + 1] + '. '
        else:
            break
    age = age_extract(text)
    gender = gender_extract(text)
    if (not age) or (not gender):
        return None
    else:
        return (age, gender)

"""
    If language is English. Filter non-English candidate.
"""
def en_filter(case):
    count = 0
    for char in case:
        if ord(char) >= 128:
            count += 1
    return count / len(case) <= 0.03

"""
    If length of the text is greater than 10. Filter too short candidate.
"""
def length_filter(case):
    return len(case.strip().split()) >= 10


if __name__ == "__main__":
    patient_count = 0
    patient_in_case_count = 0

    low_length_count = 0
    not_en_count = 0
    no_demo_count = 0
    new_data = []
    patients = set()
    
    data = json.load(open("../../../meta_data/patient_note_candidates.json", "r"))
    for dat in tqdm(data):
        # Remove duplicates
        if dat['patient'] in patients:
            continue
        patients.add(dat['patient'])
        # Length filter
        if not length_filter(dat['patient']):
            low_length_count += 1
            continue
        # Language filter
        if not en_filter(dat['patient']):
            not_en_count += 1
            continue
        # Demographic filter
        if not demo_filter(dat['patient']):
            no_demo_count += 1
            continue
        # Extract demographic characteristics
        age, gender = demo_filter(dat['patient'])
        temp = dat
        temp['age'] = age
        temp['gender'] = gender
        temp['patient_id'] = str(len(new_data))
        new_data.append(temp)
        patient_count += 1
        
        if dat['article_type'].strip() == 'case-report':
            patient_in_case_count += 1

    print("Patient count: ", patient_count)
    print("Patient in case report type count: ", patient_in_case_count)
    print("Length lt 10 count: ", low_length_count)
    print("Not English count: ", not_en_count)
    print("No demographic count: ", no_demo_count)
    print("PMC-Patients count: ", len(new_data))

    patients = []
    PMIDs = []
    # Generate patient_uid
    for i in range(len(new_data)):
        patient = new_data[i]
        PMCID = patient['file_path'].split('/')[-1][3:-4]
        if (i == 0) or (patients[i - 1]['patient_uid'].split("-")[0] != PMCID):
            index = "1"
        elif (i > 0) and (patients[i - 1]['patient_uid'].split("-")[0] == PMCID):
            index = str(int(patients[i - 1]['patient_uid'].split('-')[1]) + 1)
        patient_uid = PMCID + '-' + index
        temp = {"patient_id": patient["patient_id"], "patient_uid": patient_uid, "PMID": patient['PMID'], "file_path": patient['file_path'],\
            "title": patient['title'], "patient": patient['patient'], "age": patient["age"], "gender": patient['gender']}
        PMIDs.append(temp['PMID'])
        patients.append(temp)

    json.dump(patients, open("../../../meta_data/PMC-Patients.json", "w"), indent = 4)
    json.dump(list(set(PMIDs)), open("../../../meta_data/PMIDs.json", "w"), indent = 4)
    
    #import ipdb; ipdb.set_trace()

