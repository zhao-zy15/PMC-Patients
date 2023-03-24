import json
import numpy as np
import torch
import os
import faiss
from tqdm import trange
from transformers import AutoTokenizer
from model import BiEncoder
from beir.retrieval.evaluation import EvaluateRetrieval


def generate_patient_embeddings(tokenizer, model, patients, device, output_dir = None, model_max_length = 512, batch_size = 500):
    with open("../../../../../datasets/patient2article_retrieval/PAR_test_qrels.tsv", "r") as f:
        lines = f.readlines()
    test_set = set([line.split('\t')[0] for line in lines[1:]])
    test_patient_uids = []
    test_patients = []
    for patient in test_set:
        test_patient_uids.append(patient)
        test_patients.append(patients[patient]['patient'])

    model.eval()
    with torch.no_grad():
        tokenized = tokenizer(test_patients, max_length = model_max_length, padding = "max_length", truncation = True, return_tensors = 'pt')
        test_embeddings = model.module.encoder_1(input_ids = tokenized['input_ids'][:batch_size].to(device), \
            attention_mask = tokenized["attention_mask"][:batch_size].to(device), \
            token_type_ids = tokenized["token_type_ids"][:batch_size].to(device))
        if output_dir:    
            test_embeddings = test_embeddings[1].detach().cpu().numpy()
        else:
            test_embeddings = test_embeddings.last_hidden_state[:, 0, :].detach().cpu().numpy()
        for i in trange(1, (len(test_patients) // batch_size)):
            temp = model.module.encoder_1(input_ids = tokenized['input_ids'][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                attention_mask = tokenized["attention_mask"][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                token_type_ids = tokenized["token_type_ids"][(i * batch_size) : ((i+1) * batch_size)].to(device))
            if output_dir:    
                temp = temp[1].detach().cpu().numpy()
            else:
                temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
            test_embeddings = np.concatenate((test_embeddings, temp), axis = 0)
        temp = model.module.encoder_1(input_ids = tokenized['input_ids'][((i+1) * batch_size):].to(device), \
            attention_mask = tokenized["attention_mask"][((i+1) * batch_size):].to(device), \
            token_type_ids = tokenized["token_type_ids"][((i+1) * batch_size):].to(device))
        if output_dir:    
            temp = temp[1].detach().cpu().numpy()
        else:
            temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
        test_embeddings = np.concatenate((test_embeddings, temp), axis = 0)
        print(test_embeddings.shape)

    if output_dir:
        np.save(os.path.join(output_dir, "test_embeddings.npy"), test_embeddings)
        json.dump(test_patient_uids, open(os.path.join(output_dir, "test_patient_uids.json"), "w"), indent = 4)

    return test_embeddings, test_patient_uids


def generate_article_embeddings(tokenizer, model, corpus, device, output_dir = None, model_max_length = 512, batch_size = 500):
    PMIDs = list(corpus.keys())
    articles = [corpus[PMID]['title'] + tokenizer.sep_token + corpus[PMID]['text'] for PMID in PMIDs]

    model.eval()
    with torch.no_grad():
        tokenized = tokenizer(articles[:batch_size], max_length = model_max_length, padding = "max_length", truncation = True, return_tensors = 'pt')
        article_embeddings = model.module.encoder_2(input_ids = tokenized['input_ids'].to(device), \
            attention_mask = tokenized["attention_mask"].to(device), \
            token_type_ids = tokenized["token_type_ids"].to(device))
        if output_dir:    
            article_embeddings = article_embeddings[1].detach().cpu().numpy()
        else:
            article_embeddings = article_embeddings.last_hidden_state[:, 0, :].detach().cpu().numpy()
        for i in trange(1, (len(articles) // batch_size)):
            tokenized = tokenizer(articles[(i * batch_size) : ((i+1) * batch_size)], max_length = model_max_length, padding = "max_length", truncation = True, return_tensors = 'pt')
            temp = model.module.encoder_2(input_ids = tokenized['input_ids'].to(device), \
                attention_mask = tokenized["attention_mask"].to(device), \
                token_type_ids = tokenized["token_type_ids"].to(device))
            if output_dir:    
                temp = temp[1].detach().cpu().numpy()
            else:
                temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
            article_embeddings = np.concatenate((article_embeddings, temp), axis = 0)
        tokenized = tokenizer(articles[((i+1) * batch_size):], max_length = model_max_length, padding = "max_length", truncation = True, return_tensors = 'pt')
        temp = model.module.encoder_2(input_ids = tokenized['input_ids'].to(device), \
            attention_mask = tokenized["attention_mask"].to(device), \
            token_type_ids = tokenized["token_type_ids"].to(device))
        if output_dir:    
            temp = temp[1].detach().cpu().numpy()
        else:
            temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
        article_embeddings = np.concatenate((article_embeddings, temp), axis = 0)
        print(article_embeddings.shape)

    if output_dir:
        np.save(os.path.join(output_dir, "article_embeddings.npy"), article_embeddings)
        json.dump(PMIDs, open(os.path.join(output_dir, "PAR_PMIDs.json"), "w"), indent = 4)

    return article_embeddings, PMIDs


def dense_retrieve(queries, query_ids, documents, doc_ids, nlist = 1024, m = 24, nprobe = 100):
    dim = queries.shape[1]

    k = 1000
    quantizer = faiss.IndexFlatIP(dim)
    # Actually for PPR, it is possible to perform exact search.
    index = faiss.IndexFlatIP(dim)
    #index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    #index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)

    print(index.is_trained)
    index.train(documents)
    print(index.is_trained)
    index.add(documents)
    index.nprobe = nprobe
    print(index.ntotal)  

    qrels = {}
    with open("../../../../../datasets/patient2article_retrieval/PAR_test_qrels.tsv", "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        q, doc, _ = line.split('\t')
        if q in qrels:
            qrels[q][doc] = 1
        else:
            qrels[q] = {doc: 1}

    print("Begin search...")
    results = index.search(queries, k)
    print("End search!")

    retrieved = {}
    for i in range(results[1].shape[0]):
        result_ids = [doc_ids[idx] for idx in results[1][i]]
        result_scores = results[0][i]
        retrieved[query_ids[i]] = {result_ids[j]: float(result_scores[j]) for j in range(k)}

    #json.dump(retrieved, open("../PAR_Dense_test.json", "w"), indent = 4)
    evaluation = EvaluateRetrieval()
    metrics = evaluation.evaluate(qrels, retrieved, [10, 1000])
    mrr = evaluation.evaluate_custom(qrels, retrieved, [index.ntotal], metric="mrr")
    return mrr[f'MRR@{index.ntotal}'], metrics[3]['P@10'], metrics[1]["MAP@10"], metrics[0]['NDCG@10'], metrics[2]['Recall@1000']


def run_metrics(output_dir):
    test_embeddings = np.load("{}/test_embeddings.npy".format(output_dir))
    article_embeddings = np.load("{}/article_embeddings.npy".format(output_dir))
    test_patient_uids = json.load(open("{}/test_patient_uids.json".format(output_dir), "r"))
    PMIDs = json.load(open("{}/PMIDs.json".format(output_dir), "r"))
    results = dense_retrieve(test_embeddings, test_patient_uids, article_embeddings, PMIDs)
    print(results)
    return


def run_unsupervised(model_name_or_path):    
    torch.distributed.init_process_group(backend = "nccl", init_method = 'env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(local_rank, device)

    model = BiEncoder(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank], output_device = local_rank)

    patients = json.load(open("../../../../../datasets/PMC-Patients.json", "r"))
    patients = {patient['patient_uid']: patient for patient in patients}
    test_embeddings, test_patient_uids = generate_patient_embeddings(tokenizer, model, patients, device)
    corpus_file = "../../../../../datasets/patient2article_retrieval/PAR_corpus.jsonl"
    corpus = []
    with open(corpus_file, "r") as f:
        for line in f:
            corpus.append(json.loads(line))
    corpus = {article['_id']: {"title": article['title'], "text": article['text']} for article in corpus}
    article_embeddings, PMIDs = generate_article_embeddings(tokenizer, model, corpus, device)
    results = dense_retrieve(test_embeddings, test_patient_uids, article_embeddings, PMIDs)
    print(results)
    return


if __name__ == "__main__":
    #model_name_or_path = "michiyasunaga/BioLinkBERT-base"
    #model_name_or_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    model_name_or_path = "emilyalsentzer/Bio_ClinicalBERT"
    #model_name_or_path = "allenai/specter"

    output_dir = "output_linkbert"

    #run_metrics(output_dir)
    run_unsupervised(model_name_or_path)

    
