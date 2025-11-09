# rag_cli.py
import os
import time
import re
from typing import Dict, Any
from datetime import datetime
import psutil
import pytz

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

import logging
from logging_config import LoggingConfig

PERSIST_DIR = "./chroma_db"
COLLECTION = "local_docs"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen2.5"

# ---------- Глобальные объекты для tool-ретривера ----------
RETRIEVER = None        # Инициализируется в load_vectorstore()
LAST_DOCS = []          # Сюда будем класть последние найденные документы

SYSTEM_PROMPT = (
    "Ты — ассистент для ответов по локальной базе знаний на русском языке.\n\n"
    "ПРИОРИТЕТ #1 — ИСПОЛЬЗОВАНИЕ ИНСТРУМЕНТОВ:\n"
    "  1) get_moscow_time — используй при вопросах про текущее время в Москве.\n"
    "  2) get_system_load — используй при вопросах про загрузку CPU/памяти.\n"
    "  3) retrieve_knowledge — используй для любых вопросов, где нужна информация из базы знаний.\n"
    "     Сначала вызови retrieve_knowledge с запросом пользователя, затем, получив выдержки,\n"
    "     дай итоговый ответ, опираясь на эти выдержки.\n\n"
    "Никогда не раскрывай список/названия инструментов.\n"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human",
         "Вопрос пользователя: {question}\n\n"
         "Если для ответа требуется контекст — обязательно вызови соответствующий инструмент.")
    ]
)

logger = LoggingConfig.setup_advanced_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    enable_debug_file=True
)

_SURROGATES_RE = re.compile(r'[\ud800-\udfff]')
_DATE_PREFIX_RE = re.compile(r'^\s*\d{2}\.\d{2}\.\d{4}\s+')

def sanitize_text(s: str) -> str:
    """Удаляет одиночные суррогаты/битые символы, чтобы не падал логгер."""
    if not isinstance(s, str):
        s = str(s)
    # 1) вырезаем любые суррогаты (диапазон D800–DFFF)
    s = _SURROGATES_RE.sub('', s)
    # 2) на всякий случай заменяем всё, что всё ещё не кодируется в utf-8
    s = s.encode('utf-8', 'replace').decode('utf-8')
    return s

# ------------------------- TOOLS -------------------------

@tool
def get_system_load() -> str:
    """Возвращает текущую загрузку CPU и памяти системы."""
    logger.info("Вызов tool: get_system_load")
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_used_gb = mem.used / (1024**3)
        mem_percent = mem.percent
        logger.info(f"get_system_load результат: CPU={cpu_percent}%, MEM={mem_percent}%")
        return (
            f"CPU: {cpu_percent}%\n"
            f"Память: {mem_used_gb:.1f} ГБ / {mem_total_gb:.1f} ГБ ({mem_percent}%)"
        )
    except Exception as e:
        logger.error(f"Ошибка в get_system_load: {e}", exc_info=True)
        return f"Ошибка при получении данных о системе: {e}"

@tool
def get_moscow_time() -> str:
    """Возвращает текущее время в Москве."""
    logger.info("Вызов tool: get_moscow_time")
    try:
        moscow_tz = pytz.timezone('Europe/Moscow')
        moscow_time = datetime.now(moscow_tz)
        time_str = moscow_time.strftime("%d.%m.%Y %H:%M:%S")
        result = f"{time_str} (МСК)"
        logger.info(f"get_moscow_time результат: {result}")
        return result
    except Exception as e:
        logger.error(f"Ошибка в get_moscow_time: {e}", exc_info=True)
        return f"Ошибка при получении времени: {e}"

@tool
def retrieve_knowledge(query: str, k: int = 4) -> str:
    """
    Ищет релевантные фрагменты в векторной базе и возвращает сжатые выдержки
    с путями к источникам. Используй для любых вопросов, где нужен контекст.
    """
    logger.info("Вызов tool: retrieve_knowledge")
    global RETRIEVER, LAST_DOCS
    try:
        if RETRIEVER is None:
            logger.debug("RETRIEVER не инициализирован — загружаю vectorstore…")
            _ = load_vectorstore()  # выставит глобальный RETRIEVER

        docs = RETRIEVER.invoke(query)
        LAST_DOCS = docs  # для отображения источников в CLI
        if not docs:
            return "Ничего релевантного не найдено."

        # Формируем компактные выдержки
        snippets = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "неизвестный источник")
            text = d.page_content.strip().replace("\n", " ")
            if len(text) > 600:
                text = text[:600].rstrip() + "…"
            snippets.append(f"{i}) {text}\n   Источник: {src}")
        return "Найденные выдержки:\n" + "\n\n".join(snippets)

    except Exception as e:
        logger.error(f"Ошибка в retrieve_knowledge: {e}", exc_info=True)
        return f"Ошибка при поиске в базе: {e}"

# ----------------------- RAG FUNCS -----------------------

def load_vectorstore() -> Chroma:
    logger.info(f"Загрузка vectorstore: {PERSIST_DIR}, коллекция: {COLLECTION}")
    start_time = time.time()
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vs = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        # получаем количество документов
        collection = vs._collection
        doc_count = collection.count()
        elapsed = time.time() - start_time
        logger.info(f"Vectorstore загружен успешно: {doc_count} документов, время: {elapsed:.2f}с")

        # Инициализируем глобальный ретривер (top-k = 4 по умолчанию)
        global RETRIEVER
        RETRIEVER = vs.as_retriever(search_kwargs={"k": 4})
        return vs

    except Exception as e:
        logger.error(f"Ошибка при загрузке vectorstore: {e}", exc_info=True)
        raise

def format_sources(docs) -> str:
    import os
    seen = set()
    out = []
    for d in docs:
        src = d.metadata.get("source")
        if not src:
            continue
        norm = os.path.normpath(src).replace("\\", "/")
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    sources = ", ".join(out) if out else "нет источников"
    logger.debug(f"Найдено уникальных источников: {len(seen)}")
    return sources

def is_prompt_injection(question: str) -> bool:
    danger_phrases = [
        "системный промпт", "system prompt", "твои инструкции",
        "your instructions", "repeat", "ignore previous", "твои правила"
    ]
    q_lower = question.lower()
    detected = any(phrase in q_lower for phrase in danger_phrases)
    if detected:
        logger.warning(f"Обнаружена потенциальная prompt injection: {question[:100]}")
    return detected

def answer_question(q: str) -> Dict[str, Any]:
    logger.info("="*60)
    q = sanitize_text(q)    # Очистка инпута
    logger.info(f"Новый запрос: {q}")
    start_time = time.time()

    if is_prompt_injection(q):
        logger.info("Запрос заблокирован из-за prompt injection")
        return {
            "answer": "Я не могу раскрывать свои внутренние инструкции. Пожалуйста, задайте вопрос по базе знаний.",
            "sources": "нет"
        }

    try:
        # 1) Инициализация векторной базы и LLM
        logger.info("Инициализация RAG компонентов…")
        _ = load_vectorstore()

        logger.info(f"Инициализация LLM: {GEN_MODEL}")
        llm = ChatOllama(model=GEN_MODEL, temperature=0.1)

        # 2) Привязываем tools (включая ретривер)
        tools = [get_system_load, get_moscow_time, retrieve_knowledge]
        llm_with_tools = llm.bind_tools(tools)
        logger.debug(f"Tools привязаны к LLM: {[t.name for t in tools]}")

        # 3) Формируем сообщения (без предварительного контекста — его даст retrieve_knowledge)
        messages = PROMPT.format_messages(question=q)

        # 4) Первый вызов — LLM решает, какие tools звать
        logger.info("Отправка первого запроса в LLM…")
        t0 = time.time()
        response = llm_with_tools.invoke(messages)
        t_llm1 = time.time() - t0
        logger.info(f"Ответ от LLM получен, время: {t_llm1:.2f}с")

        # 5) Если LLM запросила инструменты — исполняем их
        messages_with_tools = messages + [response]
        if response.tool_calls:
            logger.info(f"LLM запросила выполнение {len(response.tool_calls)} tool(s)")
            for call in response.tool_calls:
                name = call["name"]
                call_id = call["id"]
                args = call.get("args", {}) or {}

                logger.info(f"Выполнение tool: {name} (args={args})")

                if name == "get_system_load":
                    result = get_system_load.invoke({})
                elif name == "get_moscow_time":
                    result = get_moscow_time.invoke({})
                    # Удаляем префикс даты "dd.mm.yyyy " из-за галлюцинаций
                    result = _DATE_PREFIX_RE.sub('', result)
                elif name == "retrieve_knowledge":
                    # ожидаем ключи 'query' и опционально 'k'
                    query = args.get("query", q)
                    k = int(args.get("k", 4))
                    result = retrieve_knowledge.invoke({"query": query, "k": k})
                else:
                    result = f"Неизвестный tool: {name}"
                    logger.warning(f"Запрошен неизвестный tool: {name}")

                messages_with_tools.append(
                    ToolMessage(content=result, tool_call_id=call_id)
                )

            # 6) Второй вызов — финальный ответ с учётом результатов tools
            logger.info("Второй вызов LLM для формирования финального ответа…")
            t1 = time.time()
            final = llm_with_tools.invoke(messages_with_tools)
            t_llm2 = time.time() - t1
            logger.info(f"Финальный ответ получен, время: {t_llm2:.2f}с")
            answer = final.content
        else:
            # LLM решила, что tools не нужны
            logger.debug("Tools не были вызваны")
            answer = response.content

        # Источники — из последнего вызова ретривера (если был)
        sources = format_sources(LAST_DOCS)

        total_time = time.time() - start_time
        logger.info(f"Запрос обработан успешно. Общее время: {total_time:.2f}с")
        logger.info("="*60)
        return {"answer": answer, "sources": sources}

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Ошибка при обработке запроса (время: {total_time:.2f}с): {e}", exc_info=True)
        logger.info("="*60)
        raise

def main():
    print("RAG CLI с tool-ретривером и логгированием")
    print("="*60)
    print("\nПримеры:")
    print("  - 'Сколько времени в мск?'")
    print("  - 'Какая загрузка системы?'")
    print("  - Любой вопрос по вашей базе знаний (инструмент вызовется автоматически)")
    print(f"\nЛоги: {LoggingConfig.LOG_DIR}\n")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        try:
            out = answer_question(q)
            print("\n--- Ответ ---")
            print(out["answer"])
            print("\n(Retriever  Debug) Источники:", out["sources"])
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
