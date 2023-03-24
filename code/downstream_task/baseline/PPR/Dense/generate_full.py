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
        test_embeddings = model.module.encoder(input_ids = tokenized['input_ids'][:batch_size].to(device), \
            attention_mask = tokenized["attention_mask"][:batch_size].to(device), \
            token_type_ids = tokenized["token_type_ids"][:batch_size].to(device))[1].detach().cpu().numpy()
        for i in trange(1, (len(test_patients) // batch_size)):
            temp = model.module.encoder(input_ids = tokenized['input_ids'][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                attention_mask = tokenized["attention_mask"][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                token_type_ids = tokenized["token_type_ids"][(i * batch_size) : ((i+1) * batch_size)].to(device))[1].detach().cpu().numpy()
            test_embeddings = np.concatenate((test_embeddings, temp), axis = 0)
        temp = model.module.encoder(input_ids = tokenized['input_ids'][((i+1) * batch_size):].to(device), \
            attention_mask = tokenized["attention_mask"][((i+1) * batch_size):].to(device), \
            token_type_ids = tokenized["token_type_ids"][((i+1) * batch_size):].to(device))[1].detach().cpu().numpy()
        test_embeddings = np.concatenate((test_embeddings, temp), axis = 0)
        print(test_embeddings.shape)

    if output_dir:
        np.save(os.path.join(output_dir, "test_embeddings_full.npy"), test_embeddings)
        json.dump(test_patient_uids, open(os.path.join(output_dir, "test_patient_uids_full.json"), "w"), indent = 4)

    return test_embeddings, test_patient_uids


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

    print("Begin search...")
    results = index.search(queries, k)
    print("End search!")

    retrieved = {}
    for i in range(results[1].shape[0]):
        result_ids = [doc_ids[idx] for idx in results[1][i]]
        result_scores = results[0][i]
        retrieved[query_ids[i]] = {result_ids[j]: float(result_scores[j]) for j in range(k)}

    json.dump(retrieved, open("../PPR_Dense_test_full.json", "w"), indent = 4)
    return


if __name__ == "__main__":
    model_name_or_path = "michiyasunaga/BioLinkBERT-base"
    output_dir = "output_linkbert"
    
    #args = torch.load("{}/training_args.bin".format(output_dir))
    torch.distributed.init_process_group(backend = "nccl", init_method = 'env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(local_rank, device)

    model = BiEncoder(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank], output_device = local_rank)
    model.module.load_state_dict(torch.load("{}/best_model.pth".format(output_dir)))

    patients = json.load(open("../../../../../datasets/PMC-Patients.json", "r"))
    patients = {patient['patient_uid']: patient for patient in patients}
    test_embeddings, test_patient_uids = generate_embeddings(tokenizer, model, patients, device, output_dir)
    
    #test_embeddings = np.load("{}/test_embeddings_full.npy".format(output_dir))
    train_embeddings = np.load("{}/train_embeddings.npy".format(output_dir))
    #test_patient_uids = json.load(open("{}/test_patient_uids_full.json".format(output_dir), "r"))
    train_patient_uids = json.load(open("{}/train_patient_uids.json".format(output_dir), "r"))
    
    dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
