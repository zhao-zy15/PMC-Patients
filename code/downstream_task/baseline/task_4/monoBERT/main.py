import argparse
import torch
from torch import nn
from dataloader import PARDataset, MyCollateFn
from model import monoBERT, MarginMSE
import numpy as np
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader
import os
import json
import wandb
from rerank import rerank_TREC_CDS


def train(args, model, dataloader, dev_dataloader, test_dataloader, tokenizer):
    wandb.init(project="PAR_mono", entity="zhengyun21", config = args, name = args.output_dir[7:])
    dev_loss = []
    best_dev_loss = 1e10
    
    lossFn = MarginMSE(args.loss_p, args.margin, args.pos_loss_ratio)
    
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
            optimizer, num_warmup_steps = args.warmup_steps // args.gradient_accumulation_steps, num_training_steps = args.total_steps // args.gradient_accumulation_steps
        )
    if args.schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps = args.warmup_steps // args.gradient_accumulation_steps
        )
    if args.schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps = args.warmup_steps // args.gradient_accumulation_steps, num_training_steps = args.total_steps // args.gradient_accumulation_steps
        )

    global_step = 0
    pos_acc = 0
    model.train()
    # For early stopping.
    while global_step <= args.total_steps:
        bar = tqdm(dataloader)
        for _, batch in enumerate(bar):
            global_step += 1
            t_input_ids = batch[0].to(args.device)
            t_attention_mask = batch[1].to(args.device)
            t_token_type_ids = batch[2].to(args.device)
            a_input_ids = batch[3].to(args.device)
            a_attention_mask = batch[4].to(args.device)
            a_token_type_ids = batch[5].to(args.device)
            label = batch[6].to(args.device)
            _, prob = model(t_input_ids, t_attention_mask, t_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids)
            pred = []
            for score in prob:
                if score.item() < 0.5:
                    pred.append(0)
                else:
                    pred.append(1)
            pred = torch.tensor(pred).to(args.device)
            loss = lossFn(prob, label)
            pos_id = label == 1
            if any(pos_id):
                pos_acc = sum(pred[pos_id] == label[pos_id]).item() / sum(pos_id).item()
            acc = sum(pred == label).item() / label.shape[0]
            wandb.log({'loss': loss.item()}, step = global_step)
            wandb.log({'acc': acc}, step = global_step)
            wandb.log({'pos_acc': pos_acc}, step = global_step)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}".format(global_step, loss.item(), acc))

            if global_step % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                wandb.log({'lr': optimizer.state_dict()['param_groups'][0]['lr']}, step = global_step)
            
            if global_step % args.test_steps == 0:
                test_loss, test_acc, confusion = test(args, model, dev_dataloader)
                test_pos_acc = confusion[1][1] / sum(confusion[1])
                print("======Dev at step {}======".format(global_step))
                print(test_loss, test_acc)
                print(test_pos_acc)
                print(confusion)
                wandb.log({"dev_loss": test_loss}, step = global_step)
                wandb.log({"dev_acc": test_acc}, step = global_step)
                wandb.log({"dev_pos_acc": test_pos_acc}, step = global_step)
                if test_loss < best_dev_loss:
                    best_dev_loss = test_loss
                    wandb.run.summary["best_dev_loss"] = best_dev_loss
                    torch.save(model, os.path.join(args.output_dir, 'best_model.pth'))
                #if len(dev_loss) > 2 and test_loss >= dev_loss[-1] and dev_loss[-1] >= dev_loss[-2]:
                #    return
                #else:
                #    dev_loss.append(test_loss)
                torch.save(model, os.path.join(args.output_dir, 'last_model.pth'))
                predict, metrics = rerank_TREC_CDS(args, global_step, tokenizer, model)
                print(f"=====Rerank TREC at step {global_step}======")
                print(predict)
                print(metrics)
                wandb.log({"Rerank pos ratio": predict[1] / sum(predict)}, step = global_step)
                for metric in metrics:
                    wandb.log({metric: metrics[metric]}, step = global_step)
                
            

            if global_step > args.total_steps:
                return


def test(args, model, dataloader):
    model.eval()
    steps = 0
    total_loss = 0.
    confusion = [[0, 0], [0, 0]]
    lossFn = MarginMSE(args.loss_p, args.margin, args.pos_loss_ratio)
    
    with torch.no_grad():
        bar = tqdm(dataloader)
        for _, batch in enumerate(bar):
            t_input_ids = batch[0].to(args.device)
            t_attention_mask = batch[1].to(args.device)
            t_token_type_ids = batch[2].to(args.device)
            a_input_ids = batch[3].to(args.device)
            a_attention_mask = batch[4].to(args.device)
            a_token_type_ids = batch[5].to(args.device)
            label = batch[6].to(args.device)
            _, prob = model(t_input_ids, t_attention_mask, t_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids)
            pred = []
            for score in prob:
                if score.item() < 0.5:
                    pred.append(0)
                else:
                    pred.append(1)
            pred = torch.tensor(pred).to(args.device)
            pos_id = label == 1
            loss = lossFn(prob, label)
            total_loss += loss.item()
            for i in range(len(pred)):
                confusion[label[i].item()][pred[i].item()] += 1
            if any(pos_id):
                pos_acc = sum(pred[pos_id] == label[pos_id]).item() / sum(pos_id)
            else:
                pos_acc = 0.
            acc = sum(pred == label).item() / label.shape[0]
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}, Pos_acc: {:.4f}".format(steps, loss, acc, pos_acc))
            steps += 1

    model.train()
    return total_loss / steps, (confusion[0][0] + confusion[1][1]) / (sum(confusion[0]) + sum(confusion[1])), confusion


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_dir = "../../../../../datasets/task_4_patient2article_retrieval/PAR_sample"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = monoBERT(args.model_name_or_path)
    if args.para:
        model = nn.DataParallel(model)
    model.to(args.device)
    pubmed = {}
    pubmed_dir = "../../../../../../pubmed/pubmed_title_abstract/"
    for file_name in tqdm(os.listdir(pubmed_dir)):
        articles = json.load(open(os.path.join(pubmed_dir, file_name), "r"))
        for PMID in articles:
            if PMID not in pubmed or len(pubmed[PMID]['title']) * len(pubmed[PMID]['abstract']) == 0:
                pubmed[PMID] = articles[PMID]

    train_dataset = PARDataset(data_dir, "train", tokenizer, args.max_length, "both", pubmed)
    print(train_dataset.stat())
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    dev_dataset = PARDataset(data_dir, "dev", tokenizer, args.max_length, "both", pubmed)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    test_dataset = PARDataset(data_dir, "test", tokenizer, args.max_length, "both", pubmed)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.train:
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        train(args, model, train_dataloader, dev_dataloader, test_dataloader, tokenizer)
        model = torch.load(os.path.join(args.output_dir, "best_model.pth"))
        test_loss, test_acc, confusion = test(args, model, test_dataloader)
        test_pos_acc = confusion[1][1] / sum(confusion[1])
        print("======Test======")
        print(test_loss, test_acc)
        print(test_pos_acc)
        print(confusion)
        wandb.run.summary["test_acc"] = test_acc
        wandb.run.summary["test_confusion"] = confusion
        wandb.run.summary["test_pos_acc"] = test_pos_acc
        predict, metrics = rerank_TREC_CDS(args, "best", tokenizer, model)
        print(f"=====Rerank TREC Final======")
        print(predict)
        print(metrics)
        wandb.run.summary["Rerank pos ratio"] = predict[1] / sum(predict)
        for metric in metrics:
            wandb.run.summary[metric] = metrics[metric]
    else:
        checkpoint = os.path.join(args.output_dir, 'best_model.pth')
        model = torch.load(checkpoint)
        model.to(args.device)
        '''
        test_loss, test_acc, confusion = test(args, model, test_dataloader)
        test_pos_acc = confusion[1][1] / sum(confusion[1])
        print("======Test======")
        print(test_loss, test_acc)
        print(test_pos_acc)
        print(confusion)
        '''
        predict, metrics = rerank_TREC_CDS(args, "best", tokenizer, model)
        print(f"=====Rerank TREC Final======")
        print(predict)
        print(metrics)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    default = "dmis-lab/biobert-v1.1",
    type = str,
    help = "Model name or path."
)
parser.add_argument(
    "--batch_size",
    default = 24,
    type = int,
    help = "Batch size."
)
parser.add_argument(
    "--max_length",
    default = 512,
    type = int,
    help = "Max lengths for patient note and article title+abstract."
)
parser.add_argument(
    "--learning_rate",
    default = 1e-5,
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
    default = 100000,
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
    default = 5000,
    type = int,
    help = "Warmup steps."
)
parser.add_argument(
    "--test_steps",
    default = 10000,
    type = int,
    help = "Number of steps for each test performing."
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
parser.add_argument(
    "--para",
    action = "store_true",
    help = "If parallize model."
)
parser.add_argument(
    "--pos_loss_ratio",
    default = 1,
    type = float,
    help = "Positive loss ratio."
)
parser.add_argument(
    "--loss_p",
    default = 2,
    type = float,
    help = "Exponential of loss."
)
parser.add_argument(
    "--margin",
    default = 0.1,
    type = float,
    help = "Loss margin."
)
args = parser.parse_args()
run(args)