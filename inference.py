import time
from embeddings import process_llm_response
import argparse
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama2-13b-chat')
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--repetition_penalty", type=float, default=1.15)
parser.add_argument("--split_chunk_size", type=int, default=800)
parser.add_argument("--split_overlap", type=int, default=100)
parser.add_argument("--K", type=int, default=6)
parser.add_argument("--PDFs_path", type=str, default="dataset/")
parser.add_argument("--embeddings_path", type=str, default="embeddings/")
parser.add_argument("--output_folder", type=str, default="outputs/")
args = parser.parse_args()
config = Config(args)


def llm_ans(query, qa_chain):
    start = time.time()
    qa_chain.eval()
    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str
print(config.__dict__)
