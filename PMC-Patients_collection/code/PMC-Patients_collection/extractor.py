import xml.etree.cElementTree as ET
from multiprocessing import Pool
import os
import pandas as pd
import json
import re
from tqdm import trange, tqdm
import sys
sys.path.append("..")
from xml_utils import parse_paragraph, getTitle, getText, getSection, clean_text


"""
    Counting.
"""
def stat():
    print("Article: ", article_count)
    print("Case report type articles: ", case_report_type_count)
    print("Patient: ", patient_count)
    print("Patient in case report type articles: ", patient_in_case_count)
    print("Error: ", error_count)

"""
    Section_title_trigger, stricter, for single patient extraction
"""
def match_title(title):
    return title_pattern.match(title.lower())

"""
    Section_title_trigger, easier, for first step.
"""
def section_title_trigger(title):
    title = title.lower()
    if ("case" in title or "patient" in title or "clinical" in title or "medical" in title) \
        and "consent" not in title and "approv" not in title:
        return True
    return False

"""
    Get section and subsection names in a hiearchical pattern.
    Input:
        body element of article xml.
"""
def hier_parse(body):
    results = []
    results.append([body])
    while len(results[-1]) > 0:
        results.append([])
        for sec in results[-2]:
            for subsec in sec.iterfind('./sec'):
                results[-1].append(subsec)
    return results[1:-1]

"""
    Extractor.
    Input:
        file_path and PMID
    Output:
        several counts and patient_notes extracted.
"""
def extract(msg):
    file_path, PMID, License = msg
    article_count = 0
    case_report_type_count = 0
    patient_count = 0
    error_count = 0
    patients = []
    # Only extract from articles with at least CC BY-NC-SA License
    if License not in ["CC BY", "CC0", "CC BY-NC", "CC BY-NC-SA"]:
        return article_count, case_report_type_count, patient_count, error_count, patients
    f = False
    article_count += 1

    try:
        tree = ET.parse(os.path.join(data_dir, file_path))
        root = tree.getroot()
    except Exception as e:
        error_count += 1
        return article_count, case_report_type_count, patient_count, error_count, patients

    article_type = root.attrib['article-type']
    if article_type == 'case-report':
        case_report_type_count += 1
    body = root.find(".//body")
    article_title = root.find(".//article-meta").find(".//article-title")

    # Remove articles without body or title.
    if (body is None) or (article_title is None):
        error_count += 1
        return article_count, case_report_type_count, patient_count, error_count, patients

    article_title = getText(article_title)

    # Extract section / subsection with titles like "Case 1 xxx" or "Patient B"
    hierarchical_secs = hier_parse(body)
    for layer in range(len(hierarchical_secs)):
        if f:
            break
        for sec in hierarchical_secs[layer]:
            title = getTitle(sec)
            # Assume each section with such titles is a single patient note
            if case_1_pattern.match(title.lower()):
                patient = getSection(sec)
                if len(patient) > 0:
                    patients.append({"title": article_title, "file_path": file_path, "PMID": PMID, "patient": patient, "article_type": article_type})
                    patient_count += 1
                    f = True

    if f:
        return article_count, case_report_type_count, patient_count, error_count, patients

    # Extract paragraphs fullmatch "Case 1"
    index = []
    paras = parse_paragraph(body)
    for j in range(len(paras)):
        title = paras[j][0]
        paragraph_text = paras[j][1]
        # Section_title_trigger and "case 1" paragraph indicates multiple notes, trach the paragraph ids
        if section_title_trigger(title) and case_1_pattern.fullmatch(paragraph_text.lower()):
            index.append(j)

    if len(index) > 1:
        # The last patient note is taken till end of the section.
        last = len(paras)
        for j in range(index[-1] + 1, len(paras)):
            if paras[j][0] != paras[index[-1]][0]:
                last = j
                break
        index.append(last)
        # Multi_patients_extractor, extract texts between successive paragraph ids.
        # Note triggerring paragraphs are NOT included.
        for k in range(len(index) - 1):
            patient = ""
            for j in range(index[k] + 1, index[k + 1]):
                patient += paras[j][1] + '\n'
            patient = patient.strip()
            if len(patient) > 0:
                patients.append({"title": article_title, "file_path": file_path, "PMID": PMID, "patient": patient, "article_type": article_type})
                patient_count += 1
                f = True
    
    if f:
        return article_count, case_report_type_count, patient_count, error_count, patients

    # Extract paragraphs with "Case 1 xxx" / "The first case xx"
    index = []
    for j in range(len(paras)):
        title = paras[j][0]
        paragraph_text = paras[j][1]
        # Section_title_trigger and "Case 1:" or "The first patient" paragraph indicates multiple notes, trach the paragraph ids
        if section_title_trigger(title) and (case_1_pattern.match(paragraph_text.lower()) or first_pattern.match(paragraph_text.lower())):
            index.append(j)

    if len(index) > 1:
        # The last patient note is taken till end of the section.
        last = len(paras)
        for j in range(index[-1] + 1, len(paras)):
            if paras[j][0] != paras[index[-1]][0]:
                last = j
                break
        index.append(last)
        # Multi_patients_extractor, extract texts between successive paragraph ids.
        # Note triggering paragraphs are included
        for k in range(len(index) - 1):
            patient = ""
            for j in range(index[k], index[k + 1]):
                patient += paras[j][1] + '\n'
            patient = patient.strip()
            if len(patient) > 0:
                patients.append({"title": article_title, "file_path": file_path, "PMID": PMID, "patient": patient, "article_type": article_type})
                patient_count += 1
                f = True

    if f:
        return article_count, case_report_type_count, patient_count, error_count, patients

    # Extract section with title like "Case Report"
    for layer in range(len(hierarchical_secs)):
        if f:
            break
        for sec in hierarchical_secs[layer]:
            title = getTitle(sec)
            # No multiple patients identified, assume single note and extract whole section.
            if match_title(title.lower()):
                patient = getSection(sec)
                if len(patient) > 0:
                    patients.append({"title": article_title, "file_path": file_path, "PMID": PMID, "patient": patient, "article_type": article_type})
                    patient_count += 1
                    f = True
                    break

    return article_count, case_report_type_count, patient_count, error_count, patients


if __name__ == "__main__":
    # Section_title_trigger, such as "case report", "patient representation", etc.
    title_pattern = re.compile(r'(clinical )?((patient)|(case))(( ((illustrations?)|(report)|(descriptions?)|(information)|(details)|(discussions?)|((re)?presentation))([^a-z]|$))|$)')
    # Detect and further remove label in title such as "3.1" in "3.1 case one"
    label_pattern = re.compile(r'^[0-9]\.?[0-9]?\.?[0-9]?\.? ?')
    # Multi_patient_trigger, for paragraphs staring with "Case 1" and "The first patient", respectively
    case_1_pattern = re.compile(r'^(clinical )?((patient)|(case))( ((illustration)|(report)|(description)|(information)|(details)|(discussion)|((re)?presentation)))?.?\(?(([0-9]{1,2})|([abcde])|(i{1,3}|(i?vi?))|((one)|(two)|(three)|(four)|(five)))\)?($|[^a-z])')
    first_pattern = re.compile(r'^((the)|(our)) ((first)|(second)|(third)|(fourth)|(fifth)|(sixth)|(seventh)|(eighth)|(nineth)|(1-?st)|(2-?nd)|(3-?rd)|([456789]-?th)) ((case)|(patient))')
    # Convert several white space character into " "
    space = r"[\u3000\u2009\u2002\u2003\u00a0\u200a\xa0]"

    data_dir = "../../../../PMC_OA/"
    file_list = pd.read_csv(os.path.join(data_dir, "PMC_OA_meta.csv"))

    article_count = 0
    case_report_type_count = 0
    patient_count = 0
    patient_in_case_count = 0
    error_count = 0
    patients = []

    msgs = [(file_list['file_path'].iloc[i], str(file_list['PMID'].iloc[i]), file_list['License'].iloc[i]) for i in range(len(file_list))]
    pool = Pool(processes = 20)
    results = pool.map(extract, msgs)

    patient_id = 0
    for result in results:
        article_count += result[0]
        case_report_type_count += result[1]
        patient_count += result[2]
        patient_in_case_count += result[1] * result[2]
        error_count += result[3]
        patients += result[4]
        
    json.dump(patients, open("../../meta_data/patient_note_candidates.json", "w"), indent = 4)

    stat()
    import ipdb; ipdb.set_trace()

