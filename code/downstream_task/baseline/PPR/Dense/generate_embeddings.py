import json
import numpy as np
import torch
import os
import faiss
from tqdm import trange
from transformers import AutoTokenizer
from model import BiEncoder
from beir.retrieval.evaluation import EvaluateRetrieval


def generate_embeddings(tokenizer, model, patients, device, output_dir = None, model_max_length = 512, batch_size = 500):
    train_patient_uids = json.load(open("../../../../../meta_data/train_patient_uids.json", "r"))
    with open("../../../../../datasets/patient2patient_retrieval/PPR_test_qrels.tsv", "r") as f:
        lines = f.readlines()
    test_set = set([line.split('\t')[0] for line in lines[1:]])
    test_patient_uids = []
    test_patients = []
    for patient in test_set:
        test_patient_uids.append(patient)
        test_patients.append(patients[patient]['patient'])
    train_patients = [patients[patient]['patient'] for patient in train_patient_uids]

    model.eval()
    with torch.no_grad():
        tokenized = tokenizer(test_patients, max_length = model_max_length, padding = "max_length", truncation = True, return_tensors = 'pt')
        test_embeddings = model.module.encoder(input_ids = tokenized['input_ids'][:batch_size].to(device), \
            attention_mask = tokenized["attention_mask"][:batch_size].to(device), \
            token_type_ids = tokenized["token_type_ids"][:batch_size].to(device))
        if output_dir:    
            test_embeddings = test_embeddings[1].detach().cpu().numpy()
        else:
            test_embeddings = test_embeddings.last_hidden_state[:, 0, :].detach().cpu().numpy()
        for i in trange(1, (len(test_patients) // batch_size)):
            temp = model.module.encoder(input_ids = tokenized['input_ids'][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                attention_mask = tokenized["attention_mask"][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                token_type_ids = tokenized["token_type_ids"][(i * batch_size) : ((i+1) * batch_size)].to(device))
            if output_dir:    
                temp = temp[1].detach().cpu().numpy()
            else:
                temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
            test_embeddings = np.concatenate((test_embeddings, temp), axis = 0)
        temp = model.module.encoder(input_ids = tokenized['input_ids'][((i+1) * batch_size):].to(device), \
            attention_mask = tokenized["attention_mask"][((i+1) * batch_size):].to(device), \
            token_type_ids = tokenized["token_type_ids"][((i+1) * batch_size):].to(device))
        if output_dir:    
            temp = temp[1].detach().cpu().numpy()
        else:
            temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
        test_embeddings = np.concatenate((test_embeddings, temp), axis = 0)
        print(test_embeddings.shape)

        tokenized = tokenizer(train_patients, max_length = model_max_length, padding = "max_length", truncation = True, return_tensors = 'pt')
        train_embeddings = model.module.encoder(input_ids = tokenized['input_ids'][:batch_size].to(device), \
            attention_mask = tokenized["attention_mask"][:batch_size].to(device), \
            token_type_ids = tokenized["token_type_ids"][:batch_size].to(device))
        if output_dir:    
            train_embeddings = train_embeddings[1].detach().cpu().numpy()
        else:
            train_embeddings = train_embeddings.last_hidden_state[:, 0, :].detach().cpu().numpy()
        for i in trange(1, (len(train_patients) // batch_size)):
            temp = model.module.encoder(input_ids = tokenized['input_ids'][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                attention_mask = tokenized["attention_mask"][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                token_type_ids = tokenized["token_type_ids"][(i * batch_size) : ((i+1) * batch_size)].to(device))
            if output_dir:    
                temp = temp[1].detach().cpu().numpy()
            else:
                temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
            train_embeddings = np.concatenate((train_embeddings, temp), axis = 0)
        temp = model.module.encoder(input_ids = tokenized['input_ids'][((i+1) * batch_size):].to(device), \
            attention_mask = tokenized["attention_mask"][((i+1) * batch_size):].to(device), \
            token_type_ids = tokenized["token_type_ids"][((i+1) * batch_size):].to(device))
        if output_dir:    
            temp = temp[1].detach().cpu().numpy()
        else:
            temp = temp.last_hidden_state[:, 0, :].detach().cpu().numpy()
        train_embeddings = np.concatenate((train_embeddings, temp), axis = 0)
        print(train_embeddings.shape)

    if output_dir:
        np.save(os.path.join(output_dir, "test_embeddings.npy"), test_embeddings)
        np.save(os.path.join(output_dir, "train_embeddings.npy"), train_embeddings)
        json.dump(test_patient_uids, open(os.path.join(output_dir, "test_patient_uids.json"), "w"), indent = 4)
        json.dump(train_patient_uids, open(os.path.join(output_dir, "train_patient_uids.json"), "w"), indent = 4)

    return test_embeddings, test_patient_uids, train_embeddings, train_patient_uids


def dense_retrieve(queries, query_ids, documents, doc_ids, nlist = 1024, m = 24, nprobe = 100):
    dim = queries.shape[1]

    k = 10000
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
    with open("../../../../../datasets/patient2patient_retrieval/PPR_test_qrels.tsv", "r") as f:
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

    json.dump(retrieved, open("../PPR_Dense_test.json", "w"), indent = 4)
    evaluation = EvaluateRetrieval()
    metrics = evaluation.evaluate(qrels, retrieved, [10, 1000])
    mrr = evaluation.evaluate_custom(qrels, retrieved, [index.ntotal], metric="mrr")
    return mrr[f'MRR@{index.ntotal}'], metrics[3]['P@10'], metrics[1]["MAP@10"], metrics[0]['NDCG@10'], metrics[2]['Recall@1000']


def run_metrics(output_dir):
    test_embeddings = np.load("{}/test_embeddings.npy".format(output_dir))
    train_embeddings = np.load("{}/train_embeddings.npy".format(output_dir))
    test_patient_uids = json.load(open("{}/test_patient_uids.json".format(output_dir), "r"))
    train_patient_uids = json.load(open("{}/train_patient_uids.json".format(output_dir), "r"))
    results = dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
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
    test_embeddings, test_patient_uids, train_embeddings, train_patient_uids = generate_embeddings(tokenizer, model, patients, device)
    results = dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
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

    
