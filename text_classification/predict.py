import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .config import id2label, label2id
from transformers import logging


def load_model(model_name="FacebookAI/roberta-base", model_path="text_classification_model/roberta-base.pt",
               num_labels=23):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=23,
                                                               label2id=label2id, id2label=id2label)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, tokenizer


def predict(query, model_name="FacebookAI/roberta-base", model_path="text_classification_model/roberta-base.pt",
            num_labels=23, max_len=128):
    logging.set_verbosity_error()

    model, tokenizer = load_model(model_name, model_path, num_labels)
    inputs = tokenizer(query, return_tensors="pt", max_length=max_len, padding="max_length", truncation=True)
    inputs.to("cuda")
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = logits.cpu().numpy()
        predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]
    # return
