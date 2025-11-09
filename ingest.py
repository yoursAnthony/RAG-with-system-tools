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

def normalize_path(path_str: str) -> str:
    """
    Нормализует путь к единому формату (Unix-style с /)
    Это предотвращает дубликаты в базе из-за разных слэшей
    """
    # Преобразуем все слэши в прямые
    normalized = path_str.replace("\\", "/")
    # Убираем повторяющиеся слэши
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized

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
                # Нормализуем путь перед добавлением
                pdf_doc_list = PyPDFLoader(str(p)).load()
                for doc in pdf_doc_list:
                    doc.metadata["source"] = normalize_path(doc.metadata.get("source", str(p)))
                pdf_docs.extend(pdf_doc_list)
            except Exception:
                pass

        # Загружаем документы
        txt_docs = txt_loader.load()
        md_docs = md_loader.load()
        
        # Нормализуем пути для всех документов
        for doc in txt_docs + md_docs:
            if "source" in doc.metadata:
                doc.metadata["source"] = normalize_path(doc.metadata["source"])
        
        docs = txt_docs + md_docs + pdf_docs
    else:
        # Фолбэк без лоадеров: читаем .txt и .md вручную
        for ext in ("*.txt", "*.md"):
            for fp in path.rglob(ext):
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = fp.read_text(errors="ignore")
                # Нормализуем путь
                normalized_source = normalize_path(str(fp))
                docs.append(Document(page_content=text, metadata={"source": normalized_source}))

    return docs

def main():
    print("Загрузка документов из ./data...")
    raw_docs = load_documents_from_dir("./data")
    print(f"Загружено документов: {len(raw_docs)}")

    print("Разбиение на чанки...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],  # дефолты ок
    )
    split_docs = splitter.split_documents(raw_docs)
    print(f"Получено чанков: {len(split_docs)}")

    print(f"Инициализация embedding модели: {EMBED_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  # требует ollama + pull nomic-embed-text
    
    print(f"Создание/загрузка Chroma векторной базы...")
    # Инициализируем Chroma с персистенцией и своей коллекцией
    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print("Добавление документов в векторную базу...")
    # Можно добавить id, но Chroma сама создаст UUID
    vector_store.add_documents(split_docs)
    # persist() не обязателен в новом API — данные пишутся сразу, но вызов не повредит
    # vector_store.persist()

    print("="*60)
    print(f"✓ Ingest завершён успешно!")
    print(f"  Документов обработано: {len(split_docs)}")
    print(f"  База данных: {PERSIST_DIR}")
    print(f"  Коллекция: {COLLECTION}")
    print("="*60)
    
    # Проверка уникальности источников
    unique_sources = set()
    for doc in split_docs:
        src = doc.metadata.get("source")
        if src:
            unique_sources.add(src)
    print(f"  Уникальных источников: {len(unique_sources)}")

if __name__ == "__main__":
    main()