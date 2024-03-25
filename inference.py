import time
import argparse
from model import get_model, get_pipeline
from data_preprocess import load_data, text_split
from embeddings import generate_embeddings_from_datasets, get_retriever, process_llm_response
from prompts import generate_prompts

parser = argparse.ArgumentParser()
# 'llama2-13b-chat'  # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
parser.add_argument("--model_name", type=str, default='llama2-13b-chat')
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--repetition_penalty", type=float, default=1.15)
parser.add_argument("--split_chunk_size", type=int, default=800)
parser.add_argument("--split_overlap", type=int, default=100)
parser.add_argument("--embeddings_model_name", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--K", type=int, default=6)
parser.add_argument("--PDFs_path", type=str, default="dataset/")
parser.add_argument("--embeddings_path", type=str, default="embeddings/")
parser.add_argument("--output_folder", type=str, default="outputs/")
parser.add_argument("--question_set_path", type=str, default="questions/QuestionSet1")
parser.add_argument("--answer_path", type=str, default="answers/answers.txt")
args = parser.parse_args()


# config = Config(args)


def llm_ans(query, qa_chain):
    start = time.time()
    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str


def read_question_set(file_path):
    with open(file_path) as f:
        return f.readlines()


if __name__ == '__main__':
    tokenizer, model, max_len = get_model(args)
    model.eval()
    llm = get_pipeline(model, tokenizer, max_len, args)

    documents = load_data(args.PDFs_path)
    texts = text_split(args, documents)

    print(f"\n\nWe have created {len(texts)} chunks from {len(documents)} pages\n\n")

    PROMPT = generate_prompts()
    vectordb = generate_embeddings_from_datasets(args, texts)

    qa_chain = get_retriever(llm, vectordb, PROMPT, args)

    question_set = read_question_set(args.question_set_path)
    with open(args.answer_path, 'w') as f:
        for query in question_set:
            result = llm_ans(query, qa_chain)
            f.write(result)
            print(result)
            print("\n\n----------------------------------------------------\n\n")
