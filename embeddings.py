import argparse
import os.path
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap


def generate_embeddings_from_datasets(config: argparse.PARSER, texts):
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
        vectorDB = FAISS.load_local(f"{config.embeddings_path}/{config.embeddings_model_name}", embeddings)

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
