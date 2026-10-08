# ADR 0005: Ollama как первый local LLM runtime

- Статус: принято
- Дата: 2026-10-08

## Контекст

AirGapRAG нужен локальный LLM runtime с простым способом подготовки модели и
HTTP API. Бизнес-логика не должна зависеть от runtime, потому что позже проект
добавит llama.cpp и vLLM.

## Решение

Первым adapter становится Ollama. Приложение обращается к его локальному
`/api/chat`, но зависит только от собственного `LLMProvider`. Ollama включается
через `LLM_PROVIDER=ollama` и опциональный Compose profile. Model pull остаётся
ручным preparation step и никогда не запускается приложением.

## Альтернативы

- llama.cpp server даёт больше контроля над GGUF и параметрами запуска, но
  требует больше platform-specific настройки для первой интеграции;
- vLLM подходит для GPU throughput и concurrent serving, но слишком тяжёл для
  минимального локального runtime;
- OpenAI-compatible cloud API противоречит основной privacy boundary проекта.

## Последствия

- локальную генерацию легко запустить и диагностировать через HTTP;
- отсутствие модели обнаруживается readiness check;
- Ollama API и lifecycle изолированы внутри adapter;
- model weights и Ollama image нужно заранее включить в будущий offline bundle;
- llama.cpp и vLLM можно добавить без изменения RAG application service.
