# ingest.py
from pathlib import Path
from typing import List
import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Опционально: попробуем задействовать готовые лоадеры LangChain Community
# Если не нужны PDF — можно удалить PyPDFLoader.
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
    USE_LOADERS = True
except Exception:
    USE_LOADERS = False

PERSIST_DIR = "./chroma_db"
COLLECTION = "local_docs"
EMBED_MODEL = "nomic-embed-text"   # ollama embeddings model

def load_documents_from_dir(data_dir: str) -> List[Document]:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Папка {data_dir} не найдена")

    docs: List[Document] = []

    if USE_LOADERS:
        # Поддержка .txt .md .pdf (рекурсивно)
        # Для PDF нужен установленный pypdf (ставится вместе с PyPDFLoader зависимостями)
        txt_loader = DirectoryLoader(
            data_dir, glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},  # или {"encoding": "utf-8"}
            show_progress=True,
            use_multithreading=True
        )
        md_loader = DirectoryLoader(
            data_dir, glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},  # или {"encoding": "utf-8"}
            show_progress=True,
            use_multithreading=True
        )
        pdf_paths = list(path.rglob("*.pdf"))
        pdf_docs = []
        for p in pdf_paths:
            try:
                pdf_docs.extend(PyPDFLoader(str(p)).load())
            except Exception:
                pass

        docs = txt_loader.load() + md_loader.load() + pdf_docs
    else:
        # Фолбэк без лоадеров: читаем .txt и .md вручную
        for ext in ("*.txt", "*.md"):
            for fp in path.rglob(ext):
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = fp.read_text(errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": str(fp)}))

    return docs

def main():
    raw_docs = load_documents_from_dir("./data")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],  # дефолты ок
    )
    split_docs = splitter.split_documents(raw_docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  # требует ollama + pull nomic-embed-text
    # Инициализируем Chroma с персистенцией и своей коллекцией
    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Можно добавить id, но Chroma сама создаст UUID
    vector_store.add_documents(split_docs)
    # persist() не обязателен в новом API — данные пишутся сразу, но вызов не повредит
    # vector_store.persist()

    print(f"Ingest завершён. Документов: {len(split_docs)}. БД: {PERSIST_DIR}, коллекция: {COLLECTION}")

if __name__ == "__main__":
    main()
