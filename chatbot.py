from model import get_model, get_pipeline
from generate_embeddings import load_and_generate_embeddings_per_chapter, get_retriever, process_llm_response, \
    process_llm_response_for_chatbot, next_question_prediction

from prompts import generate_prompts
import argparse
from classify_inference import llm_ans
from text_classification.predict import predict
import gradio as gr
import time
import faiss

parser = argparse.ArgumentParser()
# 'llama2-13b-chat'  # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
parser.add_argument("--model_name", type=str, default='phi-2')
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--repetition_penalty", type=float, default=1.15)
parser.add_argument("--split_chunk_size", type=int, default=800)
parser.add_argument("--split_overlap", type=int, default=100)
parser.add_argument("--embeddings_model_name", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--K", type=int, default=3)
parser.add_argument("--PDFs_path", type=str, default="dataset_per_chapter/")
parser.add_argument("--embeddings_path", type=str, default="embeddings_per_chapter/")
parser.add_argument("--next_question_predictions_path", type=str, default="next_question_predictions/")
parser.add_argument("--number_of_question", type=int, default=5)

args = parser.parse_args()
RELATED_MATERIAL = None


def llm_ans_for_chatbot(query, qa_chain):
    start = time.time()
    llm_response = qa_chain.invoke(query)
    qa, related_material = process_llm_response_for_chatbot(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return qa + time_elapsed_str, related_material


def reduce_and_make_space(text):
    text = text.replace('\n', ' ')
    idx = text.index("Answer:")
    return text[:idx] + "\n\n" + text[idx:]


def process_query(query, history):
    global RELATED_MATERIAL
    if query == "GIVE ME THE SOURCE!":
        if RELATED_MATERIAL:
            yield RELATED_MATERIAL

        else:
            yield "There doesn't have any query before, please ask question firstly!"
    else:
        predicted_category = predict(query)
        category_message = (f"Your query belongs to **{predicted_category}**. "
                            f"Then it will redirect into **{predicted_category}** knowledge base.\n\n")
        message = category_message
        category_message = category_message + "Hold on one second."

        yield category_message

        vectordb = load_and_generate_embeddings_per_chapter(args, predicted_category)

        tokenizer, model, max_len = get_model(args)
        model.eval()
        llm = get_pipeline(model, tokenizer, max_len, args)

        PROMPT = generate_prompts()

        qa_chain = get_retriever(llm, vectordb, PROMPT, args)
        qa, related_material = llm_ans_for_chatbot(query, qa_chain)
        RELATED_MATERIAL = related_material
        qa = reduce_and_make_space(qa)
        message = message + qa

        message = message + "\n\nIf you want to find the source, just type \n```GIVE ME THE SOURCE!```\n\nYou might be interested in the following questions:\n"

        message = message + next_question_prediction(args, query)

        yield message
        history.append([query, message])


if __name__ == "__main__":
    gr.ChatInterface(process_query).launch(share=True)
