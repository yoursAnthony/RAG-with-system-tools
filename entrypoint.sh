#!/usr/bin/env bash
set -euo pipefail

# -------- настройки --------
: "${OLLAMA_HOST:=http://ollama:11434}"
: "${PULL_MODELS:=true}"              # можно выключить установкой PULL_MODELS=false
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

# По желанию — автопул моделей через HTTP API Ollama
if [ "${PULL_MODELS}" = "true" ]; then
  echo "[entrypoint] Pull ${EMBED_MODEL}"
  curl -fsS -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\":\"${EMBED_MODEL}\"}" || true

  echo "[entrypoint] Pull ${GEN_MODEL}"
  curl -fsS -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\":\"${GEN_MODEL}\"}" || true
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
