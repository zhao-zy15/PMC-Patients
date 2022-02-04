import argparse
import torch
from torch import nn
from dataloader import PatientFindingDataset, MyCollateFn
from model import LSTM_CRF, get_mask
from torch.optim import Adam
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
from utils import metric


def train(args, model, dataloader, dev_dataloader, test_dataloader):
    writer = SummaryWriter(log_dir = args.output_dir)
    
    optimizer = Adam(model.parameters(), lr = args.learning_rate)

    for epoch in range(args.total_steps):
        model.train()
        bar = tqdm(dataloader)
        for global_step, batch in enumerate(bar):
            para_ids = batch[0].to(args.device)
            tags = batch[1].to(args.device)
            length = batch[2]

            mask = get_mask(length).to(args.device)
            emission = model.forward(input_data = para_ids, input_len = length)
            loss = model.get_loss(emission = emission, labels = tags, mask = mask, device = args.device)
            pred = model.get_best_path(emission = emission, mask = mask, device = args.device)
            total_token, right_token, total_ent, pred_ent, right_ent = metric(tags, pred, length)
            acc = right_token / total_token
            precision = right_ent / pred_ent if pred_ent != 0 else 0
            recall = right_ent / total_ent
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
            writer.add_scalar('loss', loss.item(), global_step = global_step + len(bar) * epoch)
            writer.add_scalar("acc", acc, global_step = global_step + len(bar) * epoch)
            writer.add_scalar("f1", f1, global_step = global_step + len(bar) * epoch)

            loss.backward()
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}".format(global_step + len(bar) * epoch, loss.item(), acc, f1))

            #nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            
            if (epoch + 1) % args.test_steps == 0:
                test_acc, precision, recall, f1 = test(args, model, dev_dataloader)
                print("======Dev at epoch {}======".format(epoch))
                print(test_acc, precision, recall, f1)
                writer.add_scalar("dev_acc", test_acc, global_step = epoch)
                writer.add_scalar("dev_precision", precision, global_step = epoch)
                writer.add_scalar("dev_recall", recall, global_step = epoch)
                writer.add_scalar("dev_f1", f1, global_step = epoch)

    
    test_acc, precision, recall, f1 = test(args, model, test_dataloader)
    print("======Test======")
    print(test_acc, precision, recall, f1)
    writer.add_scalar("test_acc", test_acc, global_step = epoch)
    writer.add_scalar("test_precision", precision, global_step = epoch)
    writer.add_scalar("test_recall", recall, global_step = epoch)
    writer.add_scalar("test_f1", f1, global_step = epoch)


def test(args, model, dataloader):
    model.eval()
    total = np.array([0, 0, 0, 0, 0])
    with torch.no_grad():
        bar = tqdm(dataloader)
        for steps, batch in enumerate(bar):
            para_ids = batch[0].to(args.device)
            tags = batch[1].to(args.device)
            length = batch[2]

            mask = get_mask(length).to(args.device)
            emission = model.forward(input_data = para_ids, input_len = length)
            loss = model.get_loss(emission = emission, labels = tags, mask = mask, device = args.device)
            pred = model.get_best_path(emission = emission, mask = mask, device = args.device)

            total_token, right_token, total_ent, pred_ent, right_ent = metric(tags, pred, length)
            total += np.array([total_token, right_token, total_ent, pred_ent, right_ent])
            acc = total[1] / total[0]
            precision = total[4] / total[3] if total[3] != 0 else 0
            recall = total[4] / total[2]
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}".format(steps, loss.item(), acc, f1))

    model.train()
    return acc, precision, recall, f1


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    data_dir = "../../../datasets/task_1_patient_finding"
    model = LSTM_CRF()
    model.to(args.device)
    
    train_dataset = PatientFindingDataset(data_dir, "train")
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    dev_dataset = PatientFindingDataset(data_dir, "dev")
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    test_dataset = PatientFindingDataset(data_dir, "test")
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.train:
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        train(args, model, train_dataloader, dev_dataloader, test_dataloader)
        torch.save(model, os.path.join(args.output_dir, 'last_model.pth'))
    else:
        checkpoint = os.path.join(args.output_dir, 'last_model.pth')
        model = torch.load(checkpoint)
        model.to(args.device)
        test_acc, precision, recall, f1 = test(args, model, test_dataloader)
        print("======Test======")
        print(test_acc, precision, recall, f1)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size",
    default = 128,
    type = int,
    help = "Batch size."
)
parser.add_argument(
    "--learning_rate",
    default = 0.1,
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
    "--total_steps",
    default = 10,
    type = int,
    help = "Number of total steps for training."
)
parser.add_argument(
    "--test_steps",
    default = 5,
    type = int,
    help = "Number of steps for each test performing."
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


