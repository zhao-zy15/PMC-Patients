import json
import numpy as np
import torch
import os
import faiss
from tqdm import trange
from transformers import AutoTokenizer
from model import BiEncoder
import sys
sys.path.append('..')
from utils import getRR, getPrecision, getRecall


def generate_embeddings(tokenizer, model, patients, device, output_dir, model_max_length = 512, batch_size = 500):
    train_patient_uids = json.load(open("../../../../../meta_data/train_patient_uids.json", "r"))
    test_set = json.load(open("../../../../../datasets/patient2patient_retrieval/PPR_test.json", "r"))
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

        tokenized = tokenizer(train_patients, max_length = model_max_length, padding = "max_length", truncation = True, return_tensors = 'pt')
        train_embeddings = model.module.encoder(input_ids = tokenized['input_ids'][:batch_size].to(device), \
            attention_mask = tokenized["attention_mask"][:batch_size].to(device), \
            token_type_ids = tokenized["token_type_ids"][:batch_size].to(device))[1].detach().cpu().numpy()
        for i in trange(1, (len(train_patients) // batch_size)):
            temp = model.module.encoder(input_ids = tokenized['input_ids'][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                attention_mask = tokenized["attention_mask"][(i * batch_size) : ((i+1) * batch_size)].to(device), \
                token_type_ids = tokenized["token_type_ids"][(i * batch_size) : ((i+1) * batch_size)].to(device))[1].detach().cpu().numpy()
            train_embeddings = np.concatenate((train_embeddings, temp), axis = 0)
        temp = model.module.encoder(input_ids = tokenized['input_ids'][((i+1) * batch_size):].to(device), \
            attention_mask = tokenized["attention_mask"][((i+1) * batch_size):].to(device), \
            token_type_ids = tokenized["token_type_ids"][((i+1) * batch_size):].to(device))[1].detach().cpu().numpy()
        train_embeddings = np.concatenate((train_embeddings, temp), axis = 0)
        print(train_embeddings.shape)

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


    data = json.load(open("../../../../../datasets/patient2patient_retrieval/PPR_test.json", "r"))

    print("Begin search...")
    results = index.search(queries, k)
    print("End search!")

    RR, p_5, recall_1k, recall_10k = ([],[],[],[])
    for i in range(results[1].shape[0]):
        golden_list = data[query_ids[i]]
        result_ids = [doc_ids[idx] for idx in results[1][i]]
        p_5.append(getPrecision(golden_list, result_ids[:5]))
        recall_1k.append(getRecall(golden_list, result_ids[:1000]))
        RR.append(getRR(golden_list, result_ids))
        recall_10k.append(getRecall(golden_list, result_ids))

    return np.mean(RR), np.mean(p_5), np.mean(recall_1k), np.mean(recall_10k)



if __name__ == "__main__":
    model_name_or_path = "dmis-lab/biobert-v1.1"
    
    #args = torch.load("output/training_args.bin")
    torch.distributed.init_process_group(backend = "nccl", init_method = 'env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(local_rank, device)

    model = BiEncoder(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids = [local_rank], output_device = local_rank)
    model.module.load_state_dict(torch.load("output/best_model.pth"))

    patients = json.load(open("../../../../../meta_data/PMC-Patients.json", "r"))
    patients = {patient['patient_uid']: patient for patient in patients}
    test_embeddings, test_patient_uids, train_embeddings, train_patient_uids = generate_embeddings(tokenizer, model, patients, device, "./output")

    np.save("output/test_embeddings.npy", test_embeddings)
    np.save("output/train_embeddings.npy", train_embeddings)
    json.dump(test_patient_uids, open("output/test_patient_uids.json", "w"), indent = 4)
    json.dump(train_patient_uids, open("output/train_patient_uids.json", "w"), indent = 4)
    
    test_embeddings = np.load("output/test_embeddings.npy")
    train_embeddings = np.load("output/train_embeddings.npy")
    test_patient_uids = json.load(open("output/test_patient_uids.json", "r"))
    train_patient_uids = json.load(open("output/train_patient_uids.json", "r"))
    results = dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
    print(results)
