import json
from elasticsearch import Elasticsearch as ES
import xml.dom.minidom
from tqdm import tqdm


def search_patient(patient):
    query = {"query": {"match": {"patient": {"query": patient}}}}
    results = es.search(body = query, index = "patient_dev", size = 5, _source = False)
    dev_results = [(x['_id'], x['_score']) for x in results['hits']['hits']]
    results = es.search(body = query, index = "patient_test", size = 5, _source = False)
    test_results = [(x['_id'], x['_score']) for x in results['hits']['hits']]
    results = {x[0]: x[1] for x in dev_results}
    for x in test_results:
        if x[0] not in results:
            results[x[0]] = x[1]
    results = sorted(results.items(), key = lambda x: x[1], reverse = True)
    return [x[0] for x in results[:5]]


def search_article(patient):
    query = {"query": {"multi_match": {"query": patient, "type": "cross_fields", "fields": ["title^3", "abstract"]}}}
    results = es.search(body = query, index = "pubmed_title_abstract", size = 5, _source = False)
    ids = [x['_id'] for x in results['hits']['hits']]
    results = []
    for aid in ids:
        article = es.get(index = "pubmed_title_abstract", id = aid)['_source']
        article['PMID'] = aid
        results.append(article)
    return results


def test_patient(patient, i):
    test = search_patient(patient)
    results = {"target_patient": patient}
    results["similar_patients"] = []
    for pid in test:
        results["similar_patients"].append(patients[pid])
    results["relevant_articles"] = search_article(patient)
    json.dump(results, open("./results/{}.json".format(i), "w"), indent = 4)


#patient = "a 10-year-old girl presents with thrombocytopenia, hematuria and proteinuria indicating glomerulitis and decreased hearing. Multiple family members also had hearing issues"
es = ES(timeout = 1000)
patients = json.load(open("../meta_data/PMC-Patients.json", "r"))
patients = {patient["patient_uid"]: patient for patient in patients}

texts = []
dom = xml.dom.minidom.parse('topics2016.xml')
root = dom.documentElement
for topic in root.childNodes:
    if topic.nodeName != "topic":
        continue
    for description in topic.childNodes:
        if description.nodeName != "note":
            continue
        text = description.childNodes[0].data
        texts.append(text)

for i, patient in tqdm(enumerate(texts)):
    test_patient(patient, i)
    