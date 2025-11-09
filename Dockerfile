# Базовый образ
FROM python:3.11-slim

# Ускоряем Python и pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Системные зависимости (минимально необходимые)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash tini \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Скопируем только requirements и установим зависимости — быстрее сборка
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Скопируем проект
COPY . /app

RUN chmod +x /app/entrypoint.sh

# Папки для персистентных данных (смонтируем volume'ами)
RUN mkdir -p /app/data /app/chroma_db /app/logs

# Окружение: говорим LangChain, где Ollama
ENV OLLAMA_HOST=http://ollama:11434

# Небольшой entrypoint со "сторожком" (tini) для аккуратного завершения
ENTRYPOINT ["/usr/bin/tini", "--"]

# Скрипт старта: подождать Ollama, прогнать ingest при необходимости и запустить CLI
CMD ["bash", "-lc", "/app/entrypoint.sh"]
