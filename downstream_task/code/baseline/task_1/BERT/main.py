import argparse
import torch
from torch import nn
from dataloader import PatientFindingDataset, MyCollateFn
from model import BERT_classify
import numpy as np
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import metric


def train(args, model, dataloader, dev_dataloader, test_dataloader):
    writer = SummaryWriter(log_dir = args.output_dir)
    lossFn = nn.CrossEntropyLoss()
    
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

    lr_scheduler = CosineAnnealingLR(optimizer, 50)

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
    while global_step <= args.total_steps:
        bar = tqdm(dataloader)
        for _, batch in enumerate(bar):
            global_step += 1
            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            token_type_ids = batch[2].to(args.device)
            tags = batch[3].to(args.device)
            _, prob = model(input_ids, attention_mask, token_type_ids)
            loss = lossFn(prob, tags)
            pred = torch.argmax(prob, axis = 1)
            acc = sum(pred == tags).item() / tags.shape[0]

            writer.add_scalar('loss', loss.item(), global_step = global_step)
            writer.add_scalar("acc", acc, global_step = global_step)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}".format(global_step, loss.item(), acc))

            if global_step % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                lr_scheduler.step()
            
            if global_step % args.test_steps == 0:
                token_acc, confusion, precision, recall, f1 = test(args, model, dev_dataloader)
                print("======Dev at step {}======".format(global_step))
                print(token_acc, precision, recall, f1)
                print(confusion)
                writer.add_scalar("dev_acc", token_acc, global_step = global_step)

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f'model_{global_step}.pth')
                torch.save(model, save_path)

    
    token_acc, confusion, precision, recall, f1 = test(args, model, test_dataloader)
    print("======Test======")
    print(token_acc, precision, recall, f1)
    print(confusion)
    writer.add_scalar("test_acc", token_acc, global_step = global_step)


def test(args, model, dataloader):
    model.eval()
    steps = 0
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    total = np.array([0, 0, 0, 0, 0])
    with torch.no_grad():
        bar = tqdm(dataloader)
        article_tag = []
        article_pred = []
        for _, batch in enumerate(bar):
            steps += 1
            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            token_type_ids = batch[2].to(args.device)
            tags = batch[3].to(args.device)
            index = batch[4]
            _, prob = model(input_ids, attention_mask, token_type_ids)
            pred = torch.argmax(prob, axis = 1)
            for i in range(pred.shape[0]):
                confusion[tags[i].item() - 1][pred[i].item() - 1] += 1
                if index[i] != 0 or len(article_pred) == 0:
                    article_tag.append(tags[i].item())
                    article_pred.append(pred[i].item())
                else:
                    total_token, right_token, total_ent, pred_ent, right_ent = metric(article_tag, article_pred)
                    total += np.array([total_token, right_token, total_ent, pred_ent, right_ent])
                    token_acc = total[1] / total[0]
                    precision = total[4] / total[3] if total[3] != 0 else 0
                    recall = total[4] / total[2]
                    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
                    article_pred = [pred[i].item()]
                    article_tag = [tags[i].item()]
                        
            bar.set_description("Step: {}, Acc: {:.4f}, F1: {:.4f}".format(steps, token_acc, f1))

    model.train()
    return token_acc, confusion, precision, recall, f1


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    data_dir = "../../../../datasets/task_1_patient_note_recognition"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BERT_classify(args.model_name_or_path)
    model.to(args.device)
    
    train_dataset = PatientFindingDataset(data_dir, tokenizer, "train", 512)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    dev_dataset = PatientFindingDataset(data_dir, tokenizer, "dev", 512)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)
    test_dataset = PatientFindingDataset(data_dir, tokenizer, "test", 512)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.train:
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        train(args, model, train_dataloader, dev_dataloader, test_dataloader)
        torch.save(model, os.path.join(args.output_dir, 'last_model.pth'))
        tokenizer.save_pretrained(args.output_dir)
    else:
        checkpoint = os.path.join(args.output_dir, 'last_model.pth')
        model = torch.load(checkpoint)
        model.to(args.device)
        test_dataset = PatientFindingDataset(data_dir, tokenizer, "test", 512)
        test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)
        token_acc, confusion, precision, recall, f1 = test(args, model, test_dataloader)
        print("======Test======")
        print(token_acc, precision, recall, f1)
        print(confusion)


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
    "--learning_rate",
    default = 5e-5,
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
    default = 2000,
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
    default = 100,
    type = int,
    help = "Warmup steps."
)
parser.add_argument(
    "--test_steps",
    default = 1000,
    type = int,
    help = "Number of steps for each test performing."
)
parser.add_argument(
    "--save_steps",
    default = 1000,
    type = int,
    help = "Number of steps for each checkpoint."
)
parser.add_argument(
    "--schedule", 
    type=str, 
    default="cosine",
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