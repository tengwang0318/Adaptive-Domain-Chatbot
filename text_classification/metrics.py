from sklearn.metrics import f1_score, recall_score, precision_score
import torch
from tqdm import tqdm


def calculate_metrics(model, data_loader):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to('cuda')
            attention_mask = batch["attention_mask"].to('cuda')
            labels = batch["labels"].to('cuda')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
        f1 = f1_score(true_labels, predictions, average="weighted")
        precision = precision_score(true_labels, predictions, average="weighted")
        recall = recall_score(true_labels, predictions, average="weighted")
    return precision, recall, f1
