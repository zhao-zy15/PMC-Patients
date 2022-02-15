from multiprocessing import Pool
import xml.etree.cElementTree as ET
import pandas as pd
import json
import re
from tqdm import trange, tqdm
import os
import sys
sys.path.append("../../../PMC-Patients_collection/code")
from xml_utils import parse_paragraph, clean_text, getSection

""" 
    Find index of given patient note in an article.
    Input:
        paras: results of parse_paragraph
        patient: patient note
    Output:
        (start, end): index of starting and endding paragraph of the note.
"""
def match_patient_note(paras, patient_note):
    for i in range(len(paras)):
        para = paras[i][1]
        # Single paragraph matching patient note.
        if para == patient_note:
            return (i, i + 1)
        # Multiple paragraphs matching patient note.
        # Note full match is required.
        if patient_note.startswith(para):
            temp = para
            cur = i + 1
            while cur < len(paras) and paras[cur][1] in patient_note:
                temp += '\n' + paras[cur][1]
                cur += 1
                if temp == patient_note:
                    return (i, cur)
    assert True, "No matching."

"""
    Generate PNR instances for single note in single article.
    Input: 
        patient note.
    Output:
        PNR instances.
"""
def ins_generate(patient):
    file_path = patient['file_path']
    PMID = patient['PMID']
    patient_note = patient['patient']
    title = patient['title']
    tree = ET.parse(os.path.join(directory, file_path))
    root = tree.getroot()
    if root.find(".//abstract") is None:
        abstract = ""
    else:
        abstract = getSection(root.find(".//abstract"))
    body = root.find(".//body")
    paras = parse_paragraph(body)
    # Match index of starting and ending paragraphs of a patient
    start, end = match_patient_note(paras, patient_note)
    instance = {"PMID": PMID, "file_path": file_path, "title": title, "abstract": abstract, "texts": []}
    f = False
    for i in range(len(paras)):
        instance["texts"].append(paras[i][1])
    # Annotate BIO tags
    tags = 'O' * start + 'B' + 'I' * (end - start - 1) + 'O' * (len(paras) - end)

    instance['tags'] = [char for char in tags]
    # Check for validity of generated tags sequence.
    BIOES = r"O*BI*O*"
    assert re.fullmatch(BIOES, tags), "Invalid tags of article " + PMID
    
    return instance


if __name__ == "__main__": 
    # Directory of PMC-Patients   
    patients = json.load(open("../../../PMC-Patients_collection/meta_data/PMC-Patients.json", "r"))
    directory = "../../../../PMC_OA"

    # For each note, generate one instance, and merge instances of same article later.
    pool = Pool(processes = 10)
    instances = pool.map(ins_generate, patients)

    # PMID2instance dict.
    PMID2ins = {}
    for ins in instances:
        if ins['PMID'] in PMID2ins.keys():
            PMID2ins[ins['PMID']].append(ins)
        else:
            PMID2ins[ins['PMID']] = [ins]
    del instances

    PNR_dataset = []
    # Merge instances of same article and generate final dataset.
    for PMID in PMID2ins:
        if len(PMID2ins[PMID]) == 1:
            PNR_dataset.append(PMID2ins[PMID][0])
        else:
            temp = PMID2ins[PMID][0]
            for i in range(1, len(PMID2ins[PMID])):
                for j in range(len(temp['tags'])):
                    if PMID2ins[PMID][i]['tags'][j] != 'O':
                        temp['tags'][j] = PMID2ins[PMID][i]['tags'][j]
            PNR_dataset.append(temp)

    print(len(PNR_dataset))
    json.dump(PNR_dataset, open("../../meta_data/patient_note_recognition.json", "w"), indent = 4)

