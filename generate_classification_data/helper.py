import argparse
import os.path
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# completely copy from another file, due to bad design of me
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
        vectorDB = FAISS.load_local(f"{config.embeddings_path}/{config.embeddings_model_name}", embeddings, embeddings,
                                    allow_dangerous_deserialization=True)

    return vectorDB

def load_data(PDFs_path):
    loader = DirectoryLoader(PDFs_path,
                             glob=f"*.pdf",
                             loader_cls=PyPDFLoader,
                             show_progress=True,
                             use_multithreading=True
                             )
    return loader.load()


def text_split(config, documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.split_chunk_size,
        chunk_overlap=config.split_overlap
    )
    texts = text_splitter.split_documents(documents)
    return texts