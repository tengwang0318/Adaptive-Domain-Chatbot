import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import calculate_metrics
from config import id2label, label2id
from dataset import DifferentChapterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-base")
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--file_path", type=str, default="question_label_data_pair.csv")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--output_path", type=str, default="text_classification_model")
args = parser.parse_args()
if __name__ == "__main__":
    df = pd.read_csv(args.file_path, header=None)
    texts = df[0].values
    labels = df[1].values
    # print(texts)
    # print(labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=23,
                                                               label2id=label2id, id2label=id2label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.to(device)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, train_size=0.9)
    train_dataset = DifferentChapterDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer,
                                            max_len=args.max_len)

    valid_dataset = DifferentChapterDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer,
                                            max_len=args.max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[4, 6, 8, 10], gamma=0.5)
    best_f1_score = 0
    best_model_state = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}")

        for step, batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'batch_loss': loss.item()})

        scheduler.step()
        average_loss = total_loss / len(train_dataloader)
        print("average_loss: {}".format(average_loss))

        precision, recall, f1 = calculate_metrics(model, valid_dataloader)
        print(f"precision: {precision}\nrecall: {recall}\nf1: {f1}")
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_state = model.state_dict()
    torch.save(best_model_state, f"{args.output_path}/{args.model_name.split('/')[1]}.pt")
