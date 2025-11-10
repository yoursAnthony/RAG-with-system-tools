#!/usr/bin/env bash
set -euo pipefail

# -------- настройки --------
: "${OLLAMA_HOST:=http://ollama:11434}"
: "${PULL_MODELS:=false}"              # можно выключить установкой PULL_MODELS=false
: "${EMBED_MODEL:=nomic-embed-text}"  # должен совпадать с EMBED_MODEL в коде
: "${GEN_MODEL:=qwen2.5}"             # должен совпадать с GEN_MODEL в коде
# ---------------------------

echo "[entrypoint] OLLAMA_HOST=${OLLAMA_HOST}"
echo "[entrypoint] EMBED_MODEL=${EMBED_MODEL}, GEN_MODEL=${GEN_MODEL}"

# Ждём, пока Ollama поднимется
echo -n "[entrypoint] Ожидание Ollama ..."
for i in {1..120}; do
  if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo " OK"
    break
  fi
  echo -n "."
  sleep 1
  if [ "$i" -eq 120 ]; then
    echo " FAIL"; exit 1
  fi
done

# По желанию — автопул моделей (используем ollama CLI для синхронности)
if [ "${PULL_MODELS}" = "true" ]; then
  echo "[entrypoint] Загрузка ${EMBED_MODEL}..."

  # Есть ли модель?
  if ! curl -fsS "${OLLAMA_HOST}/api/show" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${EMBED_MODEL}\"}" >/dev/null 2>&1; then
    echo "[entrypoint] Модель ${EMBED_MODEL} не найдена, загружаем (stream)..."
    # Стримим прогресс: каждую JSON-строчку приводим к короткому статусу
    curl -N "${OLLAMA_HOST}/api/pull" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${EMBED_MODEL}\",\"stream\":true}" \
      | jq -rc '."status" // .status // . | tostring'
    echo "[entrypoint] ${EMBED_MODEL} — готово"
  else
    echo "[entrypoint] Модель ${EMBED_MODEL} уже загружена"
  fi

  echo "[entrypoint] Загрузка ${GEN_MODEL}..."
  if ! curl -fsS "${OLLAMA_HOST}/api/show" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${GEN_MODEL}\"}" >/dev/null 2>&1; then
    echo "[entrypoint] Модель ${GEN_MODEL} не найдена, загружаем (stream)..."
    curl -N "${OLLAMA_HOST}/api/pull" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${GEN_MODEL}\",\"stream\":true}" \
      | jq -rc '."status" // .status // . | tostring'
    echo "[entrypoint] ${GEN_MODEL} — готово"
  else
    echo "[entrypoint] Модель ${GEN_MODEL} уже загружена"
  fi

  # Проверка загруженных моделей
  echo "[entrypoint] Проверка доступных моделей:"
  curl -fsS "${OLLAMA_HOST}/api/tags" | grep -o '"name":"[^"]*"' || echo "Не удалось получить список моделей"
fi

# Запуск приложения:
# 1) если есть данные, можно один раз прогнать ingest
if [ -d "/app/data" ] && [ -n "$(ls -A /app/data 2>/dev/null || true)" ]; then
  echo "[entrypoint] Ингест данных (однократно, если chroma_db пуста)"
  if [ -z "$(ls -A /app/chroma_db 2>/dev/null || true)" ]; then
    python /app/ingest.py || true
  else
    echo "[entrypoint] chroma_db уже существует — пропускаем ingest"
  fi
else
  echo "[entrypoint] Папка /app/data пуста — ingest пропущен"
fi

# 2) интерактивный CLI
echo "[entrypoint] Запуск RAG CLI"
exec python /app/rag_cli.py