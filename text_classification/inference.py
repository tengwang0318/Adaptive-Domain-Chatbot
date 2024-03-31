import argparse
import pandas as pd
import torch
from config import label2id, id2label
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

from metrics import calculate_metrics
from torch.utils.data import DataLoader
from dataset import DifferentChapterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-base")
parser.add_argument("--model_path", type=str, default="text_classification_model/roberta-base.pt")
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--file_path", type=str, default="question_label_data_pair.csv")
args = parser.parse_args()
if __name__ == "__main__":
    df = pd.read_csv(args.file_path, header=None)
    texts = df[0].values
    labels = df[1].values

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=23,
                                                               label2id=label2id, id2label=id2label)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    # print("Loaded model parameters:")
    # print(state_dict.keys())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.to(device)
    dataset = DifferentChapterDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    precision, recall, f1 = calculate_metrics(model, dataloader)
    print(f"precision: {precision}\nrecall: {recall}\nf1: {f1}")
