import locale
import gradio as gr
import time
from model import get_model, get_pipeline
from data_preprocess import load_data, text_split
from embeddings import generate_embeddings_from_datasets, get_retriever, process_llm_response
from prompts import generate_prompts
from config import Config

config = Config()


def llm_ans(query, qa_chain):
    start = time.time()
    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str


tokenizer, model, max_len = get_model(config)
model.eval()
llm = get_pipeline(model, tokenizer, max_len, config)

documents = load_data(config.PDFs_path)
texts = text_split(config, documents)

print(f"\n\nWe have created {len(texts)} chunks from {len(documents)} pages\n\n")

PROMPT = generate_prompts()
vectordb = generate_embeddings_from_datasets(config, texts)

qa_chain = get_retriever(llm, vectordb, PROMPT, config)


def predict(message, history):
    # output = message # debug mode

    output = str(llm_ans(message)).replace("\n", "<br/>")
    return output


demo = gr.ChatInterface(
    predict,
    title=f' Open-Source LLM ({config.model_name}) for Health Care'
)

demo.queue()
demo.launch()

