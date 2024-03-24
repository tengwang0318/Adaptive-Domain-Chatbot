import os
import glob
import langchain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter


# print(sorted(glob.glob('dataset/*')))
def load_data(config: Config):
    PDFs_path = config.PDFs_path
    loader = DirectoryLoader(PDFs_path,
                             glob.glob("./*.pdf"),
                             loader_cls=PyPDFLoader,
                             show_progress=True,
                             use_multithreading=True
                             )
    return loader.load()


def text_split(config: Config, documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.split_chunk_size,
        chunk_overlap=config.split_overlap
    )
    texts = text_splitter.split_documents(documents)
    return texts
