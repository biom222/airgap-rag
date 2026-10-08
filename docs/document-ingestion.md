# Document ingestion

## Граница PHASE 2

На этом этапе реализован безопасный приём файла и независимые компоненты parsing
и chunking. Очередь заданий, retries и orchestration появятся в PHASE 4. Поэтому
upload создаёт `Document` со статусом `PENDING`, но не выполняет тяжёлую работу в
HTTP request.

## Upload

1. `UploadFile` читается блоками, а не целиком.
2. Одновременно вычисляются размер и SHA-256.
3. Файл сначала записывается в `.staging` внутри document storage.
4. Проверяются имя, расширение, MIME и сигнатура формата.
5. Для DOCX проверяются обязательные ZIP entries, encryption flag и суммарный
   распакованный размер.
6. По SHA-256 ищется существующий документ.
7. Новый файл атомарно перемещается под server-generated storage key.
8. Metadata фиксируется в PostgreSQL.

Уникальный constraint по SHA-256 закрывает race двух одновременных uploads. Если
вторая транзакция проиграла гонку, её файл удаляется, а API возвращает metadata
первого документа.

## Parsing

`DocumentParser` принимает путь к уже проверенному файлу и возвращает
`ParsedDocument` со списком `ParsedPage`.

- `PdfParser` сохраняет номера страниц, начиная с 1;
- `DocxParser` извлекает paragraphs и cells таблиц, `page=None`;
- `TextParser` читает UTF-8/UTF-8 BOM, `page=None`.

Parsing не знает о chunk size, PostgreSQL или HTTP.

## Chunking

`RecursiveCharacterChunker` сначала ищет естественную границу: пустая строка,
перевод строки, конец предложения или пробел. Если границы нет, используется
жёсткий character limit.

Каждый `Chunk` содержит:

- deterministic UUID v5;
- `document_id`;
- сквозной `chunk_index`;
- исходный `page` либо `null`;
- текст;
- SHA-256 текста.

Детерминированный ID и уникальность `(document_id, chunk_index)` подготавливают
идемпотентную запись chunks в последующей worker-фазе.

## Security limits

- путь из клиентского filename запрещён;
- storage key создаёт сервер;
- разрешены только PDF, DOCX и UTF-8 TXT;
- upload и распакованный DOCX имеют независимые лимиты;
- encrypted DOCX и PDF не поддерживаются;
- содержимое не отправляется во внешние сервисы.
