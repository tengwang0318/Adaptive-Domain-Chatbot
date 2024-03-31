import argparse
import os.path
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap
from data_preprocess import load_data_in_file, text_split


def generate_embeddings_from_datasets(config: argparse.PARSER, texts=None):
    if config.embeddings_model_name == "all-MiniLM-L6-v2":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    elif config.embeddings_model_name == "bge-large-en":
        model_name = "BAAI/bge-large-en"
    elif config.embeddings_model_name == "bge-m3":
        model_name = "BAAI/bge-m3"
    else:
        raise ValueError("Wrong embedding models")

    if not os.path.exists(f'{config.embeddings_path}/{config.embeddings_model_name}/index.faiss'):
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"}
        )
        vectorDB = FAISS.from_documents(
            documents=texts,
            embedding=embeddings
        )
        vectorDB.save_local(f"{config.embeddings_path}/{config.embeddings_model_name}")

    else:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"}
        )
        vectorDB = FAISS.load_local(f"{config.embeddings_path}/{config.embeddings_model_name}", embeddings,
                                    allow_dangerous_deserialization=True)

    return vectorDB


def get_retriever(llm, vectorDB, PROMPT, config: argparse.PARSER):
    retriever = vectorDB.as_retriever(search_kwargs={"k": config.K, "search_type": "similarity"})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )
    return qa_chain


def wrap_text_preserve_newlines(text, width=700):
    lines = text.split('\n')

    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])

    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4]
            + ' - page: '
            + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )

    ans = ans + '\n\nSources: \n' + sources_used
    return ans


def process_llm_response_for_chatbot(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])

    idx = ans.index("SPLIT_END_MARKER!!!")
    length = len("SPLIT_END_MARKER!!!")
    related_material = ans[:idx]
    qa = ans[idx + length:]
    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4]
            + ' - page: '
            + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )

    related_material = related_material + '\n\nSources: \n' + sources_used
    return qa, related_material


def load_and_generate_embeddings_per_chapter(config: argparse.PARSER, predicted_category):
    if config.embeddings_model_name == "all-MiniLM-L6-v2":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    elif config.embeddings_model_name == "bge-large-en":
        model_name = "BAAI/bge-large-en"
    elif config.embeddings_model_name == "bge-m3":
        model_name = "BAAI/bge-m3"
    else:
        raise ValueError("Wrong embedding models")

    if os.path.exists(f"{config.embeddings_path}/{config.embeddings_model_name}/{predicted_category}/index.faiss"):
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"}
        )
        vectorDB = FAISS.load_local(f"{config.embeddings_path}/{config.embeddings_model_name}/{predicted_category}",
                                    embeddings, allow_dangerous_deserialization=True)
    else:
        documents = load_data_in_file(f"{config.PDFs_path}/{predicted_category}.pdf")
        texts = text_split(config, documents)
        print(
            f"\n\nWe have created {len(texts)} chunks from {len(documents)} pages for {predicted_category} category\n\n")
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"}
        )
        vectorDB = FAISS.from_documents(
            documents=texts,
            embedding=embeddings
        )
        vectorDB.save_local(f"{config.embeddings_path}/{config.embeddings_model_name}/{predicted_category}")

    return vectorDB
