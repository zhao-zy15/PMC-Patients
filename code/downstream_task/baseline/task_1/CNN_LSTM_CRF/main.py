import argparse
import torch
from torch import nn
from dataloader import PatientFindingDataset, MyCollateFn
from model import CNN_LSTM_CRF, get_mask
from transformers import AutoTokenizer, AutoModel
from torch.optim import Adam
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
import os
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.append("..")
from utils import batch_metric, batch_para_metric


def train(args, model, dataloader, dev_dataloader, test_dataloader):
    writer = SummaryWriter(log_dir = args.output_dir)
    dev_loss = []
    best_f1 = 0.
    
    optimizer = Adam(model.parameters(), lr = args.learning_rate)

    lr_scheduler = CosineAnnealingLR(optimizer, 50)

    for epoch in range(args.total_steps):
        model.train()
        bar = tqdm(dataloader)
        for global_step, batch in enumerate(bar):
            input_ids = batch[0].to(args.device)
            tags = batch[1].to(args.device)
            length = batch[2]

            mask = get_mask(length).to(args.device)
            loss = model.neg_log_likelihood(input_ids, tags.transpose(0, 1), mask, length)
            pred = model.predict(input_ids, mask, length)

            total_token, right_token, total_ent, pred_ent, right_ent = batch_metric(tags, pred, length, args.tag2id)
            acc = right_token / total_token
            precision = right_ent / pred_ent if pred_ent != 0 else 1
            recall = right_ent / total_ent
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

            writer.add_scalar('loss', loss.item(), global_step = global_step + len(bar) * epoch)
            writer.add_scalar("acc", acc, global_step = global_step + len(bar) * epoch)
            writer.add_scalar("f1", f1, global_step = global_step + len(bar) * epoch)

            loss /= len(length)
            loss.backward()
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}".format(global_step + len(bar) * epoch, loss.item(), acc, f1))

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            lr_scheduler.step()

            
        if (epoch + 1) % args.test_steps == 0:
            f1, precision, recall, f1_p, precision_p, recall_p, loss = test(args, model, dev_dataloader)
            print("======Dev at epoch {}======".format(epoch))
            print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1*100, precision*100, recall*100))
            print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1_p*100, precision_p*100, recall_p*100))
            writer.add_scalar("dev_precision", precision, global_step = epoch)
            writer.add_scalar("dev_recall", recall, global_step = epoch)
            writer.add_scalar("dev_f1", f1, global_step = epoch)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model, os.path.join(args.output_dir, 'best_model.pth'))
            if len(dev_loss) > 2 and loss > dev_loss[-1] and dev_loss[-1] > dev_loss[-2]:
                break
            else:
                dev_loss.append(loss)

    model = torch.load(os.path.join(args.output_dir, "best_model.pth"))
    f1, precision, recall, f1_p, precision_p, recall_p, loss = test(args, model, test_dataloader)
    print("======Test======")
    print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1*100, precision*100, recall*100))
    print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1_p*100, precision_p*100, recall_p*100))
    writer.add_scalar("test_precision", precision, global_step = epoch)
    writer.add_scalar("test_recall", recall, global_step = epoch)
    writer.add_scalar("test_f1", f1, global_step = epoch)


def test(args, model, dataloader):
    model.eval()
    total = np.array([0, 0, 0, 0, 0])
    para_metric = np.array([0., 0., 0.])
    data_count = 0
    total_loss = 0.
    with torch.no_grad():
        bar = tqdm(dataloader)
        for steps, batch in enumerate(bar):
            input_ids = batch[0].to(args.device)
            tags = batch[1].to(args.device)
            length = batch[2]
            data_count += len(length)

            mask = get_mask(length).to(args.device)
            loss = model.neg_log_likelihood(input_ids, tags.transpose(0, 1), mask, length)
            pred = model.predict(input_ids, mask, length)
            total_loss += loss.item()
            total += batch_metric(tags, pred, length, args.tag2id)
            para_metric += np.array(batch_para_metric(tags, pred, length, args.tag2id))
            acc = total[1] / total[0]
            bar.set_description("Step: {}, Loss: {:.4f}, Acc: {:.4f}".format(steps, loss.item(), acc))

    precision = total[4] / total[3]
    recall = total[4] / total[2]
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

    precision_p, recall_p, f1_p = para_metric / data_count

    model.train()
    return f1, precision, recall, f1_p, precision_p, recall_p, total_loss / data_count


def run(args):
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    data_dir = "../../../../../datasets/task_1_patient_note_recognition"
    model_name_or_path = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_length = 256)
    biobert = AutoModel.from_pretrained(model_name_or_path)
    args.vocab_size = tokenizer.vocab_size

    model = CNN_LSTM_CRF(args, biobert.embeddings.word_embeddings.weight)
    model.to(args.device)
    
    train_dataset = PatientFindingDataset(tokenizer, args.max_length, data_dir, "train")
    args.tag2id = train_dataset.tag2id
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, collate_fn = MyCollateFn)
    dev_dataset = PatientFindingDataset(tokenizer, args.max_length, data_dir, "dev")
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)
    if args.test_on_human:
        test_dataset = PatientFindingDataset(tokenizer, args.max_length, data_dir, "human")
    else:
        test_dataset = PatientFindingDataset(tokenizer, args.max_length, data_dir, "test")
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = False, collate_fn = MyCollateFn)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if args.train:
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        train(args, model, train_dataloader, dev_dataloader, test_dataloader)
        torch.save(model, os.path.join(args.output_dir, 'last_model.pth'))
    else:
        checkpoint = os.path.join(args.output_dir, 'best_model.pth')
        model = torch.load(checkpoint)
        model.to(args.device)
        f1, precision, recall, f1_p, precision_p, recall_p, _ = test(args, model, test_dataloader)
        print("======Test======")
        print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1*100, precision*100, recall*100))
        print("F1:{:.3f}, precision:{:.3f}, recall:{:.3f}".format(f1_p*100, precision_p*100, recall_p*100))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_length",
    default = 256,
    type = int,
    help = "Max length of a paragraph."
)
parser.add_argument(
    "--kernel_num",
    default = 128,
    type = int,
    help = "Number of kernels."
)
parser.add_argument(
    "--kernel_size",
    default = [7,8,9],
    type = list,
    help = "Sizes of kernels."
)
parser.add_argument(
    "--dropout",
    default = 0.1,
    type = float,
    help = "Dropout."
)
parser.add_argument(
    "--batch_size",
    default = 128,
    type = int,
    help = "Batch size."
)
parser.add_argument(
    "--learning_rate",
    default = 0.01,
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
    default = 2,
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
parser.add_argument(
    "--test_on_human",
    action = "store_true",
    help = "If test model on human annotations."
)
parser.add_argument(
    "--freeze",
    action = "store_true",
    help = "If freeze word vectors."
)
args = parser.parse_args()
run(args)


