import argparse
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from dataloader import PPR_BiEncoder_Dataset, MyCollateFn
from model import BiEncoder
import numpy as np
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from generate_embeddings import generate_embeddings, dense_retrieve
import os
import wandb


def train(args, model, dataloader, dev_dataloader):
    wandb.init(project="PPR_BiEncoder", entity="zhengyun", config = args, name = args.output_dir[7:])
    dev_loss = []
    best_loss = 1e6
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate)

    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = args.total_steps
        )
    if args.schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps = args.warmup_steps
        )
    if args.schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = args.total_steps
        )

    global_step = 0
    model.train()
    lossFn = nn.NLLLoss()
    ls = nn.LogSoftmax(dim = 1)
    # For early stopping.
    while global_step <= args.total_steps if args.total_steps > 0 else 1000000000000:
        bar = tqdm(dataloader)
        for _, batch in enumerate(bar):
            global_step += 1
            input_ids_1 = batch[0].to(args.device)
            attention_mask_1 = batch[1].to(args.device)
            token_type_ids_1 = batch[2].to(args.device)
            input_ids_2 = batch[3].to(args.device)
            attention_mask_2 = batch[4].to(args.device)
            token_type_ids_2 = batch[5].to(args.device)
            scores  = model(input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2, token_type_ids_2)

            softmax_scores = ls(scores)
            targets = torch.arange(scores.shape[0]).to(args.device)
            loss = lossFn(softmax_scores, targets)
            _, max_idxs = torch.max(softmax_scores, 1)
            acc = (max_idxs == targets).sum().item() / scores.shape[0]

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}".format(global_step, loss.item(), acc))
            wandb.log({'loss': loss.item()}, step = global_step)
            wandb.log({'acc': acc}, step = global_step)

            if global_step % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            if global_step % args.test_steps == 0:
                test_results = test(args, model, dev_dataloader)
                torch.distributed.all_reduce(test_results)
                loss = (test_results[2] / test_results[3]).item()
                acc = (test_results[0] / test_results[1]).item()
                if args.local_rank == 0:
                    print("======Dev at step {}======".format(global_step))
                    print(acc)
                    print(loss)
                    wandb.log({"dev_loss": loss}, step = global_step)
                    wandb.log({"dev_acc": acc}, step = global_step)
                    if loss < best_loss:
                        best_loss = loss
                        wandb.run.summary["best_dev_loss"] = best_loss
                        torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

                if len(dev_loss) > 2 and loss > dev_loss[-1] and dev_loss[-1] > dev_loss[-2]:
                    return
                else:
                    dev_loss.append(loss)

            if args.total_steps > 0 and global_step > args.total_steps:
                return


def test(args, model, dataloader):
    print("====Begin test=====")
    model.eval()
    steps = 0
    total_loss = 0.
    right = 0
    total = 0
    lossFn = nn.NLLLoss()
    ls = nn.LogSoftmax(dim = 1)
    with torch.no_grad():
        bar = tqdm(dataloader)
        for _, batch in enumerate(bar):
            input_ids_1 = batch[0].to(args.device)
            attention_mask_1 = batch[1].to(args.device)
            token_type_ids_1 = batch[2].to(args.device)
            input_ids_2 = batch[3].to(args.device)
            attention_mask_2 = batch[4].to(args.device)
            token_type_ids_2 = batch[5].to(args.device)
            scores  = model(input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2, attention_mask_2, token_type_ids_2)

            softmax_scores = ls(scores)
            targets = torch.arange(scores.shape[0]).to(args.device)
            loss = lossFn(softmax_scores, targets)
            _, max_idxs = torch.max(softmax_scores, 1)
            acc = (max_idxs == targets).sum().item() / scores.shape[0]
            right += (max_idxs == targets).sum().item()
            total += scores.shape[0]
            
            total_loss += loss.item()
            bar.set_description("Step: {}, Acc: {:.4f}".format(steps, acc))
            steps += 1

    model.train()
    return torch.Tensor([right, total, total_loss, steps]).to(args.device)


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    torch.distributed.init_process_group(backend = "nccl", init_method = 'env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    args.device = torch.device("cuda", local_rank)
    args.local_rank = local_rank
    print(local_rank, args.device)

    data_dir = "../../../../../datasets/patient2patient_retrieval"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BiEncoder(args.model_name_or_path)
    model.to(args.device)
    model = DistributedDataParallel(model, device_ids = [local_rank], output_device = local_rank)
    
    train_dataset = PPR_BiEncoder_Dataset(data_dir, "train", tokenizer)
    sampler = DistributedSampler(train_dataset, shuffle =True)
    train_dataloader = DataLoader(train_dataset, args.batch_size, collate_fn = MyCollateFn, sampler = sampler)
    dev_dataset = PPR_BiEncoder_Dataset(data_dir, "dev", tokenizer)
    sampler = DistributedSampler(dev_dataset, shuffle =True)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, collate_fn = MyCollateFn, sampler = sampler)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.train:
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        train(args, model, train_dataloader, dev_dataloader)
        if args.local_rank == 0:
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'last_model.pth'))
            tokenizer.save_pretrained(args.output_dir)

        del model
        torch.cuda.empty_cache()
        model = BiEncoder(args.model_name_or_path)
        model.to(args.device)
        model = DistributedDataParallel(model, device_ids = [local_rank], output_device = local_rank)
        model.module.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
        test_results = test(args, model, dev_dataloader)
        torch.distributed.all_reduce(test_results)
        if args.local_rank == 0:
            loss = (test_results[2] / test_results[3]).item()
            acc = (test_results[0] / test_results[1]).item()
            print("======Final_Dev======")
            print(acc)
            print(loss)
            wandb.run.summary["final_dev_loss"] = loss
            wandb.run.summary["final_dev_acc"] = acc

            test_embeddings, test_patient_uids, train_embeddings, train_patient_uids = generate_embeddings(
                tokenizer, model, train_dataset.patients, args.device, args.output_dir, train_dataset.max_length)
            results = dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
            print(results)
            wandb.run.summary['MRR'] = results[0]
            wandb.run.summary['P@5'] = results[1]
            wandb.run.summary['R@1k'] = results[2]
            wandb.run.summary['R@10k'] = results[3]

    else:
        model.module.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
        test_results = test(args, model, dev_dataloader)
        torch.distributed.all_reduce(test_results)
        if args.local_rank == 0:
            loss = (test_results[2] / test_results[3]).item()
            acc = (test_results[0] / test_results[1]).item()
            print("======Final_Dev======")
            print(acc)
            print(loss)

            torch.cuda.empty_cache()
            test_embeddings, test_patient_uids, train_embeddings, train_patient_uids = generate_embeddings(
                tokenizer, model, train_dataset.patients, args.device, args.output_dir, train_dataset.max_length)
            results = dense_retrieve(test_embeddings, test_patient_uids, train_embeddings, train_patient_uids)
            print(results)
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    default = "michiyasunaga/BioLinkBERT-base",
    #michiyasunaga/BioLinkBERT-base
    #microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    type = str,
    help = "Model name or path."
)
parser.add_argument(
    "--batch_size",
    default = 12,
    type = int,
    help = "Batch size."
)
parser.add_argument(
    "--learning_rate",
    default = 2e-5,
    type = float,
    help = "Learning rate."
)
parser.add_argument(
    "--max_grad_norm",
    default = 1.0,
    type = float,
    help = "Max gradient norm."
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type = int,
    default = 4,
    help = "Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--total_steps",
    default = 20000,
    type = int,
    help = "Number of total steps for training."
)
parser.add_argument(
    "--weight_decay",
    default = 0.05,
    type = float,
    help = "Weight decay rate."
)
parser.add_argument(
    "--warmup_steps",
    default = 2000,
    type = int,
    help = "Warmup steps."
)
parser.add_argument(
    "--test_steps",
    default = 2000,
    type = int,
    help = "Number of steps for each test performing."
)
parser.add_argument(
    "--save_steps",
    default = 100000,
    type = int,
    help = "Number of steps for each checkpoint."
)
parser.add_argument(
    "--schedule", 
    type=str, 
    default="linear",
    choices=["linear", "cosine", "constant"], 
    help="Schedule."
)
parser.add_argument(
    "--seed",
    default = 21,
    type = int,
    help = "Random seed."
)
parser.add_argument(
    "--device",
    default = "cuda:0",
    type = str,
    help = "Device of training."
)
parser.add_argument(
    "--output_dir",
    default = "output",
    type = str,
    help = "Output directory."
)
parser.add_argument(
    "--train",
    action = "store_true",
    help = "If train model."
)
args = parser.parse_args()
run(args)