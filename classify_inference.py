import os.path
import time
import argparse
from model import get_model, get_pipeline
from data_preprocess import load_data_in_folder, load_data_in_file, text_split
from generate_embeddings import process_llm_response, load_and_generate_embeddings_per_chapter, get_retriever
from prompts import generate_prompts
from text_classification.predict import predict

parser = argparse.ArgumentParser()
# 'llama2-13b-chat'  # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
parser.add_argument("--model_name", type=str, default='llama2-13b-chat')
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--repetition_penalty", type=float, default=1.15)
parser.add_argument("--split_chunk_size", type=int, default=800)
parser.add_argument("--split_overlap", type=int, default=100)
parser.add_argument("--embeddings_model_name", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--K", type=int, default=4)
parser.add_argument("--PDFs_path", type=str, default="dataset_per_chapter/")
parser.add_argument("--embeddings_path", type=str, default="embeddings_per_chapter/")

args = parser.parse_args()


def llm_ans(query, qa_chain):
    start = time.time()
    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str


if __name__ == '__main__':
    query = input("Enter your query: ")
    predicted_category = predict(query)
    print(f"You query belongs to {predicted_category}.\nIt will redirect to {predicted_category} knowledge bases.")

    vectordb = load_and_generate_embeddings_per_chapter(args, predicted_category)


    tokenizer, model, max_len = get_model(args)
    model.eval()
    llm = get_pipeline(model, tokenizer, max_len, args)

    PROMPT = generate_prompts()

    qa_chain = get_retriever(llm, vectordb, PROMPT, args)
    result = llm_ans(query, qa_chain)
    print(result)
