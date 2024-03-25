import glob
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# print(sorted(glob.glob('dataset/*.pdf')))


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


if __name__ == '__main__':
    import time

    start = time.time()
    load_data("dataset/")
    end = time.time()
    print(end - start)
