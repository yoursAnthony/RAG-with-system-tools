# RAG-система на базе Ollama и Chroma

## О решении

Это локальная (on-prem) RAG-система:
- **Индексация документов**: Chroma vector database
- **Эмбеддинги**: открытая модель `nomic-embed-text`
- **Генерация ответов**: открытая LLM `qwen2.5`
- **Оркестрация**: все модели поднимаются локально через Ollama

Такой стек полностью соответствует требованию **«on-prem OSS LLM»**: всё работает на ваших серверах/ПК без внешних API, а используемые модели — открытые и доступны в Ollama. 

В проекте это зафиксировано в коде:
```python
GEN_MODEL = "qwen2.5"
EMBED_MODEL = "nomic-embed-text"
```

Entrypoint контейнера автоматически загружает обе модели через Ollama API.

## Почему именно эти модели

### Qwen 2.5
- Современное семейство открытых моделей от Alibaba
- Доступна в вариантах для локального запуска
- Публичная документация и артефакты
- Широкая поддержка в экосистеме (в т.ч. Ollama)
- Оптимальный баланс качества/ресурсов для on-prem сценариев

    [Документация Qwen](https://github.com/QwenLM/Qwen)

### nomic-embed-text
- Открытая embedding-модель с большим контекстом (8192 токена)
- Сильные результаты на бенчмарке MTEB
- Готовый образ в Ollama
- Упрощенный локальный деплой

    [Hugging Face - nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1)

## Инструкция по сборке и запуску

**Операционная система**: Ubuntu 24.04

### Установка Docker и Docker Compose

```bash
# Устанавливаем зависимости
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Добавляем официальный GPG ключ Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Добавляем репозиторий Docker
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Устанавливаем Docker Engine и Docker Compose
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Добавляем текущего пользователя в группу docker
sudo usermod -aG docker $USER

# Применяем изменения группы (или перелогиньтесь)
newgrp docker

# Проверяем установку
docker --version
docker compose version
```

### Подготовка GPU (CUDA Toolkit)

#### Выполните последовательно следующие команды:

```bash
# Проверка наличия видеокарты
nvidia-smi

# Добавляем GPG-ключ
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
 | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Подключаем репозиторий (generic .deb), с signed-by
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
 | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
 | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# Обновляем индексы и ставим toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Привязываем рантайм к Docker и перезапускаем демон
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Проверяем появилась ли GPU в Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```



### Сборка и запуск контейнеров

В корне проекта (где находится `docker-compose.yml`):

```bash
# Сборка и запуск в фоновом режиме
docker compose up -d --build
```

Убедитесь, что контейнеры находятся в состоянии `healthy`/`running`.

### Установка моделей
```bash
# nomic-embed-text
docker exec -it ollama ollama pull nomic-embed-text
```
```bash
# qwen2.5
docker exec -it ollama ollama pull qwen2.5
```

### Первичная векторизация документов
```bash
docker exec -it rag-app python ingest.py 
```
### Запуск общения с LLM

Для входа в CLI внутри контейнера приложения:

```bash
docker exec -it rag-app python rag_cli.py
```

## Тесты
Оценка RAG с участием Judge LLM:
```bash
pytest test_llm_metrics.py -v -s
```

Тесты функциональности:
```bash
pytest test_rag_system.py -v -s
```

Тесты основанные на ключевых словах:
```bash
pytest test_metrics.py -v -s
```

---

**Удачного запуска!**
