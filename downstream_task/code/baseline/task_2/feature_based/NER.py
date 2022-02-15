import spacy
import scispacy
from scispacy.linking import EntityLinker
import json
from tqdm import tqdm
import os
import re


#spacy.prefer_gpu(1)
nlp = spacy.load("en_core_sci_scibert", exclude = ["tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

# If use GPU to run scibert, one needs to first download a spacy model and modify its max_length argument in config.
#nlp = spacy.load("scispacy_model")
linker = nlp.get_pipe("scispacy_linker")

def NER(doc):
    CUIs = []
    for entity in doc.ents:
        if entity._.kb_ents:
            CUIs.append(entity._.kb_ents[0][0])
    return CUIs


batch_size = 512
patients = json.load(open("../../../../../PMC-Patients_collection/meta_data/PMC-Patients.json", "r"))
entities = {}
texts = []
uids = []
for patient in tqdm(patients):
    uid = patient['patient_uid']
    text = patient['patient']
    uids.append(uid)
    texts.append(text)
    if len(uids) == batch_size:
        docs = nlp.pipe(texts)
        count = 0
        for doc in docs:
            CUIs = NER(doc)
            entities[uids[count]] = CUIs
            count += 1
        uids = []
        texts = []

docs = nlp.pipe(texts)
count = 0
for doc in docs:
    CUIs = NER(doc)
    entities[uids[count]] = CUIs
    count += 1

import ipdb; ipdb.set_trace()
json.dump(entities, open("NER.json", "w"), indent = 4)