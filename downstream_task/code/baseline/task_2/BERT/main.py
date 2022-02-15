import argparse
import torch
from torch import nn
from dataloader import PatientSimilarityDataset, MyCollateFn
from model import monoBERT
import numpy as np
from tqdm import trange, tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(args, model, dataloader, dev_dataloader, test_dataloader):
    writer = SummaryWriter(log_dir = args.output_dir)
    if args.mse:
        lossFn = nn.MSELoss()
    else:
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
            label = batch[3].to(args.device)
            _, prob = model(input_ids, attention_mask, token_type_ids)
            if args.mse:
                label = label - 1
                pred = []
                for score in prob:
                    if score.item() < -0.5:
                        pred.append(-1)
                    elif score.item() < 0.5:
                        pred.append(0)
                    else:
                        pred.append(1)
                pred = torch.tensor(pred).to(args.device)
                loss = lossFn(prob, label.float())
            else:
                loss = lossFn(prob, label)
                pred = torch.argmax(prob, axis = 1)
            acc = sum(pred == label).item() / label.shape[0]

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
                total, acc, test_acc, ocnfusion = test(args, model, dev_dataloader)
                print("======Dev at step {}======".format(global_step))
                print(total, acc, test_acc)
                print(confusion)
                writer.add_scalar("dev_acc", test_acc, global_step = global_step)

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f'model_{global_step}.pth')
                torch.save(model, save_path)

    
    total, acc, test_acc, confusion = test(args, model, test_dataloader)
    print("======Test======")
    print(total, acc, test_acc)
    print(confusion)
    writer.add_scalar("test_acc", test_acc, global_step = global_step)


def test(args, model, dataloader):
    model.eval()
    steps = 0
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with torch.no_grad():
        bar = tqdm(dataloader)
        total = [0, 0, 0]
        acc = [0, 0, 0]
        for _, batch in enumerate(bar):
            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            token_type_ids = batch[2].to(args.device)
            label = batch[3].to(args.device)
            _, prob = model(input_ids, attention_mask, token_type_ids)
            if args.mse:
                label = label - 1
                pred = []
                for score in prob:
                    if score.item() < -0.5:
                        pred.append(-1)
                    elif score.item() < 0.5:
                        pred.append(0)
                    else:
                        pred.append(1)
                pred = torch.tensor(pred).to(args.device)
            else:
                pred = torch.argmax(prob, axis = 1)
            for i in range(pred.shape[0]):
                confusion[label[i].item()][pred[i].item()] += 1
                total[pred[i].item()] += 1
                if label[i].item() == pred[i].item():
                    acc[label[i].item()] += 1
            bar.set_description("Step: {}, Acc: {:.4f}".format(steps, sum(acc) / sum(total)))

    model.train()
    return total, acc, sum(acc) / sum(total), confusion


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    data_dir = "../../../../datasets/task_2_patient2patient_similarity"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = monoBERT(args.model_name_or_path, args.mse)
    model.to(args.device)
    
    train_dataset = PatientSimilarityDataset(data_dir, "train", tokenizer)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    dev_dataset = PatientSimilarityDataset(data_dir, "dev", tokenizer)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    test_dataset = PatientSimilarityDataset(data_dir, "test", tokenizer)
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
        test_dataset = PatientSimilarityDataset(data_dir, "test", tokenizer)
        test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)
        total, acc, test_acc, confusion = test(args, model, test_dataloader)
        print("======Test======")
        print(total, acc, test_acc, confusion)
        test_dataset = PatientSimilarityDataset(data_dir, "test", tokenizer, True)
        test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)
        total, acc, test_acc, confusion = test(args, model, test_dataloader)
        print("======Test======")
        print(total, acc, test_acc, confusion)


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
    "--mse",
    action = "store_true",
    help = "If use MSE loss."
)
parser.add_argument(
    "--train",
    action = "store_true",
    help = "If train model."
)
args = parser.parse_args()
run(args)