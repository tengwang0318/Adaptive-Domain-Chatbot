import csv
import os.path
from .helper import generate_embeddings_from_datasets, load_data, text_split
import argparse
from .XML_parser import parser_all_questions
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_model_name", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--embeddings_path", type=str, default="embeddings")
parser.add_argument("--K", type=int, default=6)
parser.add_argument("--PDFs_path", type=str, default="dataset/")
args = parser.parse_args()


def get_category(page: int):
    if page <= 67:
        category = "Nutritional Disorders"
        idx = 1
    elif page <= 222:
        category = "Gastrointestinal Disorders"
        idx = 2
    elif page <= 309:
        category = "Hepatic and Biliary Disorders"
        idx = 3
    elif page <= 457:
        category = "Musculoskeletal and Connective Tissue Disorders"
        idx = 4
    elif page <= 588:
        category = "Ear, Nose, Throat, and Dental Disorders"
        idx = 5
    elif page <= 689:
        category = "Eye Disorders"
        idx = 6
    elif page <= 829:
        category = "Dermatologic Disorders"
        idx = 7
    elif page <= 993:
        category = "Endocrine and Metabolic Disorders"
        idx = 8
    elif page <= 1162:
        category = "Hematology and Oncology"
        idx = 9
    elif page <= 1228:
        category = "Immunology; Allergic Disorders"
        idx = 10
    elif page <= 1594:
        category = "Infectious Diseases"
        idx = 11
    elif page <= 1706:
        category = "Psychiatric Disorders"
        idx = 12
    elif page <= 1941:
        category = "Neurologic Disorders"
        idx = 13
    elif page <= 2123:
        category = "Pulmonary Disorders"
        idx = 14
    elif page <= 2338:
        category = "Cardiovascular Disorders"
        idx = 15
    elif page <= 2395:
        category = "Critical Care Medicine"
        idx = 16
    elif page <= 2579:
        category = "Genitourinary Disorders"
        idx = 17
    elif page <= 2809:
        category = "Gynecology and Obstetrics"
        idx = 18
    elif page <= 3209:
        category = "Pediatrics"
        idx = 19
    elif page <= 3295:
        category = "Geriatrics"
        idx = 20
    elif page <= 3314:
        category = "Clinical Pharmacology"
        idx = 21
    elif page <= 3494:
        category = "Injuries; Poisoning"
        idx = 22
    else:
        category = "Special Subjects"
        idx = 23

    return category, idx


if __name__ == "__main__":
    if not os.path.exists(f'{args.embeddings_path}/{args.embeddings_model_name}/index.faiss'):
        documents = load_data(args.PDFs_path)
        texts = text_split(args, documents)

        print(f"\n\nWe have created {len(texts)} chunks from {len(documents)} pages\n\n")

    vectorDB = generate_embeddings_from_datasets(args, texts=None)
    if os.path.exists("generate_classification_data/all_query.txt"):
        with open("generate_classification_data/all_query.txt") as f:
            all_questions = list(map(lambda x: x.strip(), f.readlines()))
    else:
        all_questions = parser_all_questions()
    print(f'there are {len(all_questions)} questions')
    categories, idxes = [], []
    for idx, question in enumerate(all_questions):
        counter = Counter()
        if idx % 1000 == 0:
            print(idx)
        for question_info in vectorDB.similarity_search(question, k=5):
            category, idx = get_category(question_info.metadata['page'])
            counter[(category, idx)] += 1
        most_comon_category, most_common_idx = counter.most_common(1)[0][0]
        categories.append(most_comon_category)
        idxes.append(most_common_idx)
    with open("question_label_data_pair.csv", 'w') as f:
        writer = csv.writer(f)
        for question, idx, category in zip(all_questions, idxes, categories):
            writer.writerow([question, idx, category])
