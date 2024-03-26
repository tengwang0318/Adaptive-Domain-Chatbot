from embeddings import generate_embeddings_from_datasets
import argparse
from data_preprocess import load_data, text_split
from XML_parser import parser_all_questions

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_model_name", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--embeddings_path", type=str, default="embeddings")
parser.add_argument("--K", type=int, default=6)
parser.add_argument("--PDFs_path", type=str, default="dataset/")
args = parser.parse_args()

if __name__ == "__main__":
    documents = load_data(args.PDFs_path)
    texts = text_split(args, documents)

    print(f"\n\nWe have created {len(texts)} chunks from {len(documents)} pages\n\n")

    vectorDB = generate_embeddings_from_datasets(args, texts)
    retriever = vectorDB.as_retriever(search_kwargs={"k": args.K, "search_type": "similarity"})

    all_questions = parser_all_questions()

    for question in all_questions:
        question_embedding = vectorDB.generate_embedding(question)

        results = retriever.retrieve(question_embedding)

        print(f"\nQuestion: {question}\nTop {args.K} similar chunks:")
        for idx, (chunk_id, similarity) in enumerate(results[:args.K], start=1):
            print(f"{idx}. Chunk ID: {chunk_id}, Similarity: {similarity}")
