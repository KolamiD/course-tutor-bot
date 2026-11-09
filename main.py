"""
Course Tutor QnA Bot
---------------------
RAG-бот-репетитор по курсу для Telegram, с:
  - памятью о диалоге (переформулировка вопроса с учётом истории);
  - гибридным поиском (BM25 + векторный) и порогом релевантности;
  - структурно-осознанным чанкованием;
  - мультиязычной embedding-моделью (intfloat/multilingual-e5-large);
  - персистентной векторной базой (не переиндексируется без надобности);
  - асинхронным запуском (бот не блокируется на время индексации);
  - защитой всех LLM-вызовов через try/except + таймаут (graceful degradation);
  - модерацией ответов: порог релевантности + grounding-check вторым вызовом LLM;
  - автоматическим разбиением длинных ответов на несколько сообщений
    (лимит Telegram — 4096 символов);
  - структурированными JSON-логами.

Запуск (продакшн):
    python main.py --config configs/settings.ini

Запуск (исследовательская версия, Google Colab):
    см. notebooks/course_tutor_bot_colab.ipynb
"""

import os
import sys
import json
import glob
import time
import hashlib
import logging
import asyncio
import configparser
from pathlib import Path

# --------------------------------------------------------------------------
# Workaround: в некоторых окружениях (например, Google Colab на момент
# написания) предустановленный torchaudio собран под другую версию CUDA, чем
# torch, и падает уже при простом "import torchaudio" (RuntimeError о
# несовпадении версий). Проблема в том, что transformers безусловно
# импортирует torchaudio при загрузке ЛЮБОЙ модели — через цепочку
# modeling_layers -> processing_utils -> audio_utils, хотя аудио-функции
# нам вообще не нужны.
#
# Первая версия фикса подменяла torchaudio в sys.modules только для
# текущего процесса — но vLLM при загрузке модели запускает ОТДЕЛЬНЫЕ
# подпроцессы (для безопасной инспекции архитектуры модели), которые не
# наследуют in-memory sys.modules родителя и падают на том же импорте
# заново. Поэтому заглушку кладём физически на диск и добавляем её путь в
# PYTHONPATH — переменные окружения подпроцессы наследуют, а значит увидят
# и путь к заглушке, и она перекроет настоящий (сломанный) torchaudio при
# поиске модуля в sys.path.
def _ensure_torchaudio_importable():
    import tempfile

    stub_root = os.path.join(tempfile.gettempdir(), "_torchaudio_stub")
    stub_pkg_dir = os.path.join(stub_root, "torchaudio")
    os.makedirs(stub_pkg_dir, exist_ok=True)
    init_path = os.path.join(stub_pkg_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(
                "# Пустая заглушка torchaudio.\n"
                "# См. _ensure_torchaudio_importable() в main.py — обход конфликта\n"
                "# версий CUDA между torch и torchaudio в некоторых окружениях (Colab).\n"
            )

    # Заглушка должна перекрывать настоящий (сломанный) torchaudio при поиске
    # модуля — кладём её путь В НАЧАЛО sys.path/PYTHONPATH.
    if stub_root not in sys.path:
        sys.path.insert(0, stub_root)

    existing_pp = os.environ.get("PYTHONPATH", "")
    parts = existing_pp.split(os.pathsep) if existing_pp else []
    if stub_root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([stub_root] + parts) if parts else stub_root

    # Если настоящий (рабочий или уже упавший) torchaudio успел
    # закэшироваться в sys.modules раньше — вычищаем, чтобы следующий
    # import в этом же процессе тоже подхватил заглушку.
    for mod_name in list(sys.modules):
        if mod_name == "torchaudio" or mod_name.startswith("torchaudio."):
            del sys.modules[mod_name]

    try:
        import torchaudio  # noqa: F401
        print(f"[main.py] torchaudio-заглушка подключена ({stub_root}), включая PYTHONPATH для подпроцессов vLLM.")
    except Exception as e:
        print(f"[main.py] Не удалось подключить даже заглушку torchaudio: {e}")


_ensure_torchaudio_importable()

# --------------------------------------------------------------------------
# Workaround: vLLM запускает движок (EngineCore) в отдельном подпроцессе.
# Если CUDA уже была затронута в родительском процессе (например, в
# долгоживущей Jupyter/Colab-сессии, где до этого уже пробовали загрузить
# модель), подпроцесс, созданный через стандартный для Linux 'fork', не
# может повторно инициализировать CUDA — падает с "Cannot re-initialize
# CUDA in forked subprocess". У vLLM есть автоопределение этой ситуации
# (переключение на 'spawn'), но эта эвристика срабатывает не всегда
# надёжно (например, при повторных запусках в одной и той же сессии).
# Выставляем 'spawn' сами, явно и заранее, не полагаясь на автоопределение —
# это официально документированный способ (см. текст самой ошибки vLLM,
# ссылающийся на https://docs.vllm.ai/en/latest/usage/troubleshooting.html).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# --------------------------------------------------------------------------
# Структурированное (JSON) логирование
# --------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Простой JSON-форматтер без внешних зависимостей."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Дополнительные поля, переданные через logger.info(..., extra={...})
        for key in ("event", "user_id", "duration_ms", "error"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("tutor_bot")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("logs_out.jsonl", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------

def load_config(path: str = "settings.ini") -> dict:
    config = configparser.ConfigParser()
    config.read(path)

    def get(section, key, default, cast=str):
        try:
            value = config[section][key]
            return cast(value) if cast is not bool else (value == "True")
        except Exception:
            logger.warning(
                "Не нашёл %s.%s в конфиге, использую значение по умолчанию (%s)",
                section, key, default,
                extra={"event": "config_default"},
            )
            return default

    cfg = {
        # --- LLM ---
        "model": get("Model", "model", "t-tech/T-lite-it-1.0"),
        "max_new_tokens": get("Model", "max_new_tokens", 768, int),
        "top_k": get("Model", "top_k", 50, int),
        "top_p": get("Model", "top_p", 0.9, float),
        "temperature": get("Model", "temperature", 0.5, float),
        "gpu_memory_utilization": get("Model", "gpu_memory_utilization", 0.1, float),
        "use_swap_memory": get("Model", "use_swap_memory", False, bool),
        "swap_space_size": get("Model", "swap_space_size", "64GiB"),
        "llm_timeout_seconds": get("Model", "llm_timeout_seconds", 30, int),
        # Квантование весов (пусто = bf16/fp16 как есть). На GPU с < 16 ГБ VRAM
        # (например, бесплатный Colab T4) 8B-модель в bf16 не помещается —
        # используйте "bitsandbytes" для 4-битной загрузки.
        "quantization": get("Model", "quantization", ""),
        "max_model_len": get("Model", "max_model_len", 0, int),  # 0 = дефолт модели

        # --- Материалы курса ---
        "materials_path": get("Path_to_materials", "path", "MATERIALS"),

        # --- RAG / чанкование ---
        "rag_top_k": get("RAG", "top_k", 6, int),
        "rag_chunk_size": get("RAG", "chunk_size", 1200, int),
        "rag_chunk_overlap": get("RAG", "chunk_overlap", 200, int),

        # --- Гибридный поиск ---
        "bm25_weight": get("Hybrid", "bm25_weight", 0.4, float),
        "vector_weight": get("Hybrid", "vector_weight", 0.6, float),
        "relevance_threshold": get("Hybrid", "relevance_threshold", 0.35, float),

        # --- Векторная база ---
        "persist_directory": get("VectorStore", "persist_directory", "chroma_db"),
        "embedding_model": get(
            "VectorStore", "embedding_model", "intfloat/multilingual-e5-large"
        ),
        # "" = автоопределение (cuda если доступна, иначе cpu). Индексация
        # быстрее на GPU — по умолчанию используем её для первичного
        # построения индекса.
        "embedding_device": get("VectorStore", "embedding_device", ""),
        # Размер батча при эмбеддинге чанков (влияет на пиковую память и
        # частоту логов прогресса при индексации).
        "embedding_batch_size": get("VectorStore", "embedding_batch_size", 16, int),
        # Если True — после построения индекса embedding-модель переносится
        # с GPU на CPU, освобождая VRAM под LLM. Нужно на GPU с малым
        # объёмом памяти (например, бесплатный Colab T4), где embeddings и
        # LLM одновременно на GPU не помещаются. На сервере с достаточной
        # VRAM можно оставить False — тогда embed_query на каждый вопрос
        # тоже будет выполняться на GPU (немного быстрее).
        "free_gpu_after_indexing": get("VectorStore", "free_gpu_after_indexing", False, bool),

        # --- Память о диалоге ---
        "history_max_turns": get("Memory", "history_max_turns", 4, int),

        # --- Прочее ---
        "course_name": get("Course", "name", "курс"),
    }
    return cfg


# --------------------------------------------------------------------------
# Хэш папки с материалами — чтобы понимать, нужна ли переиндексация
# --------------------------------------------------------------------------

def compute_materials_hash(materials_path: str) -> str:
    """Хэш от списка файлов + их размеров + времени изменения.
    Меняется, если добавили/удалили/отредактировали PDF."""
    h = hashlib.sha256()
    for pdf in sorted(glob.glob(os.path.join(materials_path, "*.pdf"))):
        stat = os.stat(pdf)
        h.update(pdf.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Подготовка учебных материалов -> чанки (структурно-осознанное чанкование)
# --------------------------------------------------------------------------

def build_documents(cfg: dict):
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    materials_path = cfg["materials_path"]
    pdfs = glob.glob(os.path.join(materials_path, "*.pdf"))
    if not pdfs:
        logger.warning(
            "В папке %s не найдено ни одного PDF-файла", materials_path,
            extra={"event": "no_pdfs_found"},
        )

    logger.info("Подгружаю материалы курса", extra={"event": "loading_materials"})
    loader = DirectoryLoader(materials_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    # Раньше все \n грубо выбрасывались — это склеивало заголовки с текстом
    # и ломало списки. Теперь схлопываем только повторяющиеся переносы,
    # сохраняя структуру абзацев.
    for doc in docs:
        text = doc.page_content
        text = "\n".join(line.strip() for line in text.splitlines())
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        doc.page_content = text

    logger.info(
        "Разбиваю тексты на чанки (размер: %s, пересечение: %s)",
        cfg["rag_chunk_size"], cfg["rag_chunk_overlap"],
        extra={"event": "chunking"},
    )
    # Разделители по убыванию приоритета: сначала пытаемся резать по
    # границам абзацев/заголовков, и только если абзац слишком длинный —
    # опускаемся до предложений и слов. Это меньше рвёт определения и
    # списки посередине по сравнению с чистым "резать по N символов".
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["rag_chunk_size"],
        chunk_overlap=cfg["rag_chunk_overlap"],
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    splits = splitter.split_documents(docs)

    # Отбрасываем чанки-огрызки (заголовок без содержания, обрывок в 1-2 слова) —
    # такие чанки только засоряют retriever и вредят /quiz-подобным сценариям.
    splits = [s for s in splits if len(s.page_content.strip()) >= 80]

    logger.info(
        "Итого чанков после фильтрации: %d", len(splits),
        extra={"event": "chunking_done"},
    )

    if not splits:
        # Явная, понятная ошибка вместо криптичного падения глубоко внутри
        # токенизатора (embed_documents([]) -> IndexError) на следующем шаге.
        # Типичная причина — забыли загрузить PDF в MATERIALS/ перед запуском.
        logger.error(
            "После обработки PDF получилось 0 чанков — материалов для "
            "индексации нет. Добавьте хотя бы один PDF с текстом в %s и "
            "перезапустите.", materials_path,
            extra={"event": "no_chunks_after_split"},
        )
        sys.exit(
            f"В папке {materials_path} нет ни одного PDF с извлекаемым "
            "текстом. Загрузите материалы курса и запустите бота заново."
        )

    return splits


# --------------------------------------------------------------------------
# Эмбеддинги: multilingual-e5 (лучше работает с русским языком, чем MiniLM)
# --------------------------------------------------------------------------

class E5Embeddings:
    """Прямая реализация через transformers.AutoModel/AutoTokenizer
    (mean-pooling + нормализация) — способ, рекомендованный самой карточкой
    модели intfloat/multilingual-e5-*.

    Раньше здесь использовалась обёртка sentence-transformers, но она тянет
    тяжёлую цепочку импортов (вплоть до peft/torchaudio), которая на
    некоторых окружениях (например, в Google Colab с несовпадающими версиями
    CUDA у torch/torchaudio) падает с ошибкой импорта, не связанной с самой
    embedding-моделью вообще. Прямой AutoModel/AutoTokenizer этой цепочки
    не задевает и работает где угодно, где есть torch + transformers.

    Модели семейства E5 требуют префиксов "query: " / "passage: " перед
    текстом — это часть их обучающего протокола, без префиксов качество
    поиска заметно падает.
    """

    def __init__(self, model_name: str, max_length: int = 512, device: str = None, batch_size: int = 16):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # device можно явно задать через конфиг (VectorStore.embedding_device).
        # По умолчанию — GPU, если доступна: разовая индексация всех чанков
        # заметно быстрее на GPU, чем на CPU (163 чанка одним батчем на CPU
        # могут считаться много минут без единого промежуточного лога).
        # Если GPU тесная (например, Colab T4 вместе с vLLM) — переносим
        # модель на CPU методом to_cpu() СРАЗУ ПОСЛЕ индексации, перед
        # загрузкой LLM (см. build_pipeline_background); дальше на CPU
        # выполняются только одиночные embed_query по одному вопросу за
        # раз — это быстро в любом случае.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def to_cpu(self):
        """Переносит модель на CPU и освобождает VRAM. Вызывается после
        построения индекса, если нужно освободить GPU под LLM."""
        if self.device == "cpu":
            return
        self.model.to("cpu")
        self.device = "cpu"
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        logger.info("Embedding-модель перенесена на CPU, VRAM освобождена", extra={"event": "embeddings_moved_to_cpu"})

    def _average_pool(self, last_hidden_states, attention_mask):
        mask = attention_mask[..., None].bool()
        last_hidden = last_hidden_states.masked_fill(~mask, 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _encode(self, texts):
        import torch.nn.functional as F

        batch = self.tokenizer(
            texts, max_length=self.max_length, padding=True,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**batch)
        embeddings = self._average_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts):
        # Батчами вместо одного гигантского вызова на все тексты сразу —
        # так меньше пиковая нагрузка на память и виден прогресс в логах
        # (без этого длинная индексация на CPU выглядит как зависание).
        results = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            results.extend(self._encode([f"passage: {t}" for t in batch_texts]))
            done = min(start + self.batch_size, total)
            logger.info(
                "Эмбеддинг чанков: %d/%d (device=%s)", done, total, self.device,
                extra={"event": "embedding_progress"},
            )
        return results

    def embed_query(self, text):
        return self._encode([f"query: {text}"])[0]


def build_embeddings(cfg: dict):
    logger.info(
        "Подгружаю embedding-модель %s", cfg["embedding_model"],
        extra={"event": "loading_embeddings"},
    )
    try:
        return E5Embeddings(
            cfg["embedding_model"],
            device=cfg.get("embedding_device") or None,
            batch_size=cfg.get("embedding_batch_size", 16),
        )
    except Exception as e:
        logger.error(
            "Не удалось подгрузить embedding-модель: %s", e,
            extra={"event": "embeddings_load_failed", "error": str(e)},
        )
        sys.exit("Ошибка в подгрузке embedding-модели")


# --------------------------------------------------------------------------
# Векторная база: персистентная, переиндексируется только если материалы изменились
# --------------------------------------------------------------------------

def build_vectorstore(all_splits, embeddings, cfg: dict):
    from langchain_chroma import Chroma

    persist_dir = cfg["persist_directory"]
    hash_file = os.path.join(persist_dir, "materials_hash.txt")
    current_hash = compute_materials_hash(cfg["materials_path"])

    reuse_existing = False
    if os.path.isdir(persist_dir) and os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            reuse_existing = True

    if reuse_existing:
        logger.info(
            "Материалы не изменились — загружаю существующий индекс из %s",
            persist_dir, extra={"event": "vectorstore_reuse"},
        )
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
            return vectorstore
        except Exception as e:
            logger.warning(
                "Не удалось загрузить существующий индекс (%s), пересобираю", e,
                extra={"event": "vectorstore_reuse_failed"},
            )

    logger.info(
        "Строю новый индекс (материалы изменились или индекса ещё нет)",
        extra={"event": "vectorstore_rebuild"},
    )
    try:
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=persist_dir,
            # По умолчанию Chroma использует L2-расстояние, из-за чего
            # similarity_search_with_relevance_scores() может возвращать
            # значения вне диапазона 0..1 (и даже отрицательные) — тогда
            # relevance_threshold работает непредсказуемо. Явно фиксируем
            # косинусную метрику, для которой relevance score корректно
            # нормализован в 0..1.
            collection_metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        logger.error(
            "Не удалось построить векторную базу: %s", e,
            extra={"event": "vectorstore_build_failed", "error": str(e)},
        )
        sys.exit("Ошибка в построении векторной базы")

    os.makedirs(persist_dir, exist_ok=True)
    with open(hash_file, "w", encoding="utf-8") as f:
        f.write(current_hash)

    return vectorstore


# --------------------------------------------------------------------------
# Гибридный поиск (BM25 + вектор) и порог релевантности
# --------------------------------------------------------------------------

def build_hybrid_retriever(vectorstore, all_splits, cfg: dict):
    from langchain_community.retrievers import BM25Retriever
    try:
        # langchain < 1.0
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        # langchain >= 1.0 вынесла EnsembleRetriever в отдельный пакет
        from langchain_classic.retrievers import EnsembleRetriever

    # BM25 ловит точные термины/аббревиатуры/цифры, которые эмбеддинг
    # иногда "смазывает" (например, номера пунктов регламента, коды курсов).
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = cfg["rag_top_k"]

    # Векторный retriever ловит смысловую близость даже при других словах.
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": cfg["rag_top_k"]})

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[cfg["bm25_weight"], cfg["vector_weight"]],
    )
    return ensemble


def is_relevant_enough(vectorstore, question: str, threshold: float) -> bool:
    """Порог релевантности: если даже лучший найденный чанк далёк от
    вопроса, значит тема, скорее всего, не в материалах курса — не стоит
    тащить его в LLM (риск и галлюцинации, и ответа "не по теме")."""
    try:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=1)
    except Exception as e:
        logger.warning(
            "Не удалось посчитать релевантность (%s) — пропускаю проверку", e,
            extra={"event": "relevance_check_failed"},
        )
        return True  # не блокируем ответ из-за технической ошибки проверки

    if not results:
        return False

    _, score = results[0]
    return score >= threshold


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

def build_llm(cfg: dict):
    from langchain_community.llms import VLLM

    logger.info("Загружаю модель %s", cfg["model"], extra={"event": "loading_llm"})

    # ВАЖНО: langchain_community.llms.VLLM — pydantic-модель со строго
    # заданным набором полей (model, trust_remote_code, temperature,
    # top_k/top_p, max_new_tokens, dtype, download_dir, vllm_kwargs, ...).
    # Параметры, которых нет среди явных полей (quantization, load_format,
    # max_model_len, gpu_memory_utilization, swap_space), передавать как
    # top-level kwargs НЕЛЬЗЯ — pydantic их молча проигнорирует (без
    # ошибки!), и они не дойдут до реального движка vLLM. Единственный
    # официальный способ пробросить произвольные аргументы конструктора
    # vllm.LLM(...) — через словарь vllm_kwargs.
    #
    # Именно так объясняется наблюдавшийся баг: в конфиге стоял
    # max_model_len=4096 и quantization=bitsandbytes, а в логах vLLM было
    # видно "Using max model len 32768" (дефолт модели) — настройки просто
    # не применялись, из-за чего модель пыталась загрузиться в полном
    # fp16 (~16 ГБ) с огромным KV-cache на GPU с 16 ГБ VRAM -> CUDA OOM
    # без внятного сообщения об ошибке.
    kwargs = dict(
        model=cfg["model"],
        trust_remote_code=True,
        max_new_tokens=cfg["max_new_tokens"],
        top_k=cfg["top_k"],
        top_p=cfg["top_p"],
        temperature=cfg["temperature"],
    )

    vllm_kwargs = {"gpu_memory_utilization": cfg["gpu_memory_utilization"]}
    if cfg["use_swap_memory"]:
        # vllm.LLM() ожидает swap_space как число ГиБ (int/float), а не
        # строку вида "64GiB" — конвертируем.
        raw = str(cfg["swap_space_size"])
        digits = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
        vllm_kwargs["swap_space"] = float(digits) if digits else 4.0
    if cfg.get("quantization"):
        # 4-битное квантование (bitsandbytes/awq/gptq) — критично для GPU
        # с малым объёмом VRAM, например бесплатного Colab T4 (16 ГБ):
        # 8B-модель в bf16 весит ~16 ГБ сама по себе, без места под KV-cache.
        vllm_kwargs["quantization"] = cfg["quantization"]
        vllm_kwargs["load_format"] = "bitsandbytes" if cfg["quantization"] == "bitsandbytes" else "auto"
    if cfg.get("max_model_len"):
        vllm_kwargs["max_model_len"] = cfg["max_model_len"]

    kwargs["vllm_kwargs"] = vllm_kwargs

    try:
        llm = VLLM(**kwargs)
    except Exception as e:
        logger.error(
            "Не удалось подгрузить модель %s: %s", cfg["model"], e,
            extra={"event": "llm_load_failed", "error": str(e)},
        )
        sys.exit("Ошибка в подгрузке модели")
    logger.info("Модель загружена", extra={"event": "llm_loaded"})
    return llm


# --------------------------------------------------------------------------
# Промпты
# --------------------------------------------------------------------------

# Шаг 1: переформулировка вопроса с учётом истории диалога ("память").
# Из "а что насчёт второго пункта?" + история делаем самостоятельный вопрос,
# который можно отправить в retriever без потери контекста.
CONDENSE_PROMPT = """
Ниже приведена история диалога студента с ботом-репетитором и новый вопрос
студента. Переформулируй новый вопрос в самостоятельный, понятный без
контекста диалога вопрос — сохрани исходный смысл, ничего не выдумывай.
Если новый вопрос уже самостоятельный и не зависит от истории — верни его
как есть, без изменений.

История диалога:
{history}

Новый вопрос: "{question}"

Самостоятельный вопрос (только текст вопроса, без пояснений):
"""

# Шаг 2: собственно ответ репетитора.
TUTOR_PROMPT = """
Ты — бот-репетитор по курсу "{course_name}". Твоя задача — помогать студенту
разобраться в материале, а не просто выдавать факты.

Правила:
- Объясняй понятия своими словами, простым языком.
- Если уместно — приводи короткий пример или аналогию.
- Если вопрос сложный — разбей объяснение на шаги или пункты.
- Отвечай СТРОГО на основе приведённых материалов курса. Если материалов
  недостаточно — честно скажи об этом, не выдумывай факты.
- Не упоминай слово "контекст" в ответе.

История диалога (для согласованности тона и терминов):
{history}

Вопрос студента: "{question}"

Материалы курса:
"{context}"
"""

# Шаг 3: grounding-check — проверяем, что ответ реально основан на контексте,
# а не на "общих знаниях" модели или на нерелевантном фрагменте, который
# случайно нашёл retriever. Это часть модерации ответов.
GROUNDING_PROMPT = """
Ниже приведены материалы курса, вопрос студента и черновой ответ бота.
Оцени: основан ли ответ ИСКЛЮЧИТЕЛЬНО на приведённых материалах, без
добавления фактов, которых там нет?

Материалы курса:
"{context}"

Вопрос: "{question}"
Черновой ответ: "{answer}"

Ответь одним словом: "ДА" — если ответ полностью основан на материалах,
"НЕТ" — если ответ содержит факты не из материалов или материалы вообще
не относятся к вопросу.
"""


# --------------------------------------------------------------------------
# Безопасный вызов LLM: try/except + таймаут на худший сценарий
# --------------------------------------------------------------------------

async def safe_llm_call(fn, *args, timeout: int, fallback, call_name: str, user_id=None, **kwargs):
    """Оборачивает блокирующий вызов LLM (chain.invoke(...)) в:
      - выполнение в отдельном потоке (чтобы не блокировать event loop);
      - таймаут — если модель зависла или перегружена;
      - перехват любых исключений — если vLLM упал, кончилась память GPU и т.п.
    При любой проблеме возвращает fallback вместо падения бота.
    """
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs), timeout=timeout
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "LLM-вызов %s выполнен успешно", call_name,
            extra={"event": "llm_call_ok", "duration_ms": duration_ms, "user_id": user_id},
        )
        return result
    except asyncio.TimeoutError:
        logger.error(
            "LLM-вызов %s превысил таймаут %ss", call_name, timeout,
            extra={"event": "llm_call_timeout", "user_id": user_id},
        )
        return fallback
    except Exception as e:
        logger.error(
            "LLM-вызов %s упал с ошибкой: %s", call_name, e,
            extra={"event": "llm_call_failed", "error": str(e), "user_id": user_id},
        )
        return fallback


# --------------------------------------------------------------------------
# Состояние бота (собирается асинхронно в фоне, чтобы не блокировать старт)
# --------------------------------------------------------------------------

class BotState:
    def __init__(self):
        self.ready = asyncio.Event()
        self.failed = False
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.cfg = None


async def build_pipeline_background(cfg: dict, state: BotState):
    """Тяжёлая инициализация (чтение PDF, эмбеддинги, LLM) выполняется в
    фоне отдельными потоками, чтобы Telegram-бот успевал отвечать на
    /start и не "висел молча" во время холодного старта."""
    try:
        all_splits = await asyncio.to_thread(build_documents, cfg)
        embeddings = await asyncio.to_thread(build_embeddings, cfg)
        vectorstore = await asyncio.to_thread(build_vectorstore, all_splits, embeddings, cfg)
        retriever = await asyncio.to_thread(build_hybrid_retriever, vectorstore, all_splits, cfg)

        if cfg.get("free_gpu_after_indexing") and hasattr(embeddings, "to_cpu"):
            # Индексация (тяжёлый батч по всем чанкам) уже позади — переносим
            # embedding-модель на CPU, чтобы вся VRAM освободилась под LLM.
            # Дальше embed_query вызывается по одному короткому вопросу за
            # раз — на CPU это быстро в любом случае.
            await asyncio.to_thread(embeddings.to_cpu)

        llm = await asyncio.to_thread(build_llm, cfg)

        state.vectorstore = vectorstore
        state.retriever = retriever
        state.llm = llm
        state.cfg = cfg
        state.ready.set()
        logger.info("Пайплайн полностью готов", extra={"event": "pipeline_ready"})
    except SystemExit:
        state.failed = True
        raise
    except Exception as e:
        state.failed = True
        logger.error(
            "Не удалось собрать пайплайн: %s", e,
            extra={"event": "pipeline_build_failed", "error": str(e)},
        )


# --------------------------------------------------------------------------
# Логика ответа на вопрос (объединяет память, retrieval, threshold, grounding)
# --------------------------------------------------------------------------

def format_history(history: list) -> str:
    if not history:
        return "(диалог только начался)"
    lines = []
    for q, a in history:
        lines.append(f"Студент: {q}")
        lines.append(f"Бот: {a}")
    return "\n".join(lines)


async def answer_question(state: BotState, user_id: int, question: str, history: list) -> str:
    cfg = state.cfg
    from langchain_core.prompts import ChatPromptTemplate

    # --- Шаг 1: переформулировка вопроса с учётом истории (память) ---
    if history:
        condense_prompt = ChatPromptTemplate.from_messages([("human", CONDENSE_PROMPT)])
        condense_chain = condense_prompt | state.llm
        standalone_question = await safe_llm_call(
            condense_chain.invoke,
            {"history": format_history(history), "question": question},
            timeout=cfg["llm_timeout_seconds"],
            fallback=question,  # если переформулировка не удалась — используем вопрос как есть
            call_name="condense_question",
            user_id=user_id,
        )
        standalone_question = (standalone_question or question).strip() or question
    else:
        standalone_question = question

    # --- Шаг 2: порог релевантности (часть модерации) ---
    relevant = await asyncio.to_thread(
        is_relevant_enough, state.vectorstore, standalone_question, cfg["relevance_threshold"]
    )
    if not relevant:
        logger.info(
            "Вопрос отфильтрован порогом релевантности", extra={"event": "below_threshold", "user_id": user_id},
        )
        return (
            "Кажется, этот вопрос не относится к материалам курса «"
            f"{cfg['course_name']}». Могу ответить только по загруженным "
            "материалам — попробуй переформулировать вопрос или уточни тему."
        )

    # --- Шаг 3: гибридный поиск (BM25 + вектор) ---
    docs = await asyncio.to_thread(state.retriever.invoke, standalone_question)
    context_text = "\n\n".join(d.page_content for d in docs) if docs else ""

    if not context_text:
        return (
            "Не нашёл в материалах курса ничего по этому вопросу. "
            "Попробуй сформулировать иначе или уточни, к какой теме это относится."
        )

    # --- Шаг 4: генерация ответа ---
    tutor_prompt = ChatPromptTemplate.from_messages([("human", TUTOR_PROMPT)])
    tutor_chain = tutor_prompt | state.llm
    fallback_answer = (
        "Извини, сейчас не получается сгенерировать ответ (перегружена модель "
        "или технический сбой). Попробуй, пожалуйста, ещё раз через минуту."
    )
    answer = await safe_llm_call(
        tutor_chain.invoke,
        {
            "course_name": cfg["course_name"],
            "history": format_history(history),
            "question": standalone_question,
            "context": context_text,
        },
        timeout=cfg["llm_timeout_seconds"],
        fallback=fallback_answer,
        call_name="generate_answer",
        user_id=user_id,
    )
    if answer == fallback_answer:
        return answer  # не имеет смысла делать grounding-check по фолбэку

    # --- Шаг 5: grounding-check (вторая часть модерации) ---
    grounding_prompt = ChatPromptTemplate.from_messages([("human", GROUNDING_PROMPT)])
    grounding_chain = grounding_prompt | state.llm
    verdict = await safe_llm_call(
        grounding_chain.invoke,
        {"context": context_text, "question": standalone_question, "answer": answer},
        timeout=cfg["llm_timeout_seconds"],
        fallback="ДА",  # если проверка не удалась технически — не блокируем ответ
        call_name="grounding_check",
        user_id=user_id,
    )
    if isinstance(verdict, str) and verdict.strip().upper().startswith("НЕТ"):
        logger.info(
            "Ответ не прошёл grounding-check, отклонён", extra={"event": "grounding_rejected", "user_id": user_id},
        )
        return (
            "Не уверен, что могу ответить на этот вопрос точно по материалам курса. "
            "Попробуй уточнить вопрос или свериться с первоисточником."
        )

    return answer.strip()


# --------------------------------------------------------------------------
# Telegram-бот
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Ограничение длины сообщений Telegram
# --------------------------------------------------------------------------

# Официальный лимит Telegram — 4096 символов на сообщение. Берём с запасом,
# т.к. лимит считается в UTF-16 code units, а не всегда ровно совпадает с
# длиной Python-строки для не-ASCII текста (кириллица, эмодзи и т.п.).
TELEGRAM_MESSAGE_LIMIT = 4000


def split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list:
    """Режет длинный ответ на несколько сообщений, укладывающихся в лимит
    Telegram. В отличие от грубой обрезки по количеству символов, старается
    резать по границам абзацев/предложений/слов — по убыванию приоритета —
    чтобы не разрывать текст посередине слова или мысли."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    parts = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = window.rfind("\n\n")
        if split_at == -1:
            split_at = window.rfind(". ")
            if split_at != -1:
                split_at += 1  # включить точку в текущую часть
        if split_at == -1:
            split_at = window.rfind(" ")
        if split_at == -1 or split_at < limit // 2:
            # Не нашлось разумной границы достаточно близко к лимиту —
            # режем жёстко, чтобы не сделать первую часть подозрительно короткой.
            split_at = limit

        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)

    return parts


async def reply_long_text(update, text: str) -> None:
    """Отправляет текст одним или несколькими сообщениями Telegram,
    в зависимости от длины (см. split_for_telegram)."""
    for chunk in split_for_telegram(text):
        await update.message.reply_text(chunk)


def get_bot_token() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token = os.environ.get("BOT_TOKEN")
    if not token:
        sys.exit(
            "Не найден BOT_TOKEN. Задайте переменную окружения BOT_TOKEN "
            "или создайте файл .env с BOT_TOKEN=<ваш_токен>."
        )
    return token


async def run_telegram_bot(cfg: dict, token: str):
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

    state = BotState()

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        context.user_data["history"] = []  # новый диалог — чистая память
        if not state.ready.is_set():
            await reply_long_text(
                update,
                f"Привет! Я бот-репетитор по курсу «{cfg['course_name']}».\n"
                "Материалы курса ещё готовятся (индексация) — как только всё будет "
                "готово, можно будет задавать вопросы.",
            )
        else:
            await reply_long_text(
                update,
                f"Привет! Я бот-репетитор по курсу «{cfg['course_name']}».\n"
                "Задай мне вопрос по материалам курса — я объясню.",
            )

    async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id if update.effective_user else None

        if not state.ready.is_set():
            if state.failed:
                await reply_long_text(
                    update,
                    "Бот сейчас недоступен из-за технической проблемы при запуске. "
                    "Попробуйте позже.",
                )
            else:
                await reply_long_text(
                    update,
                    "Материалы курса ещё готовятся, попробуй, пожалуйста, через минуту.",
                )
            return

        user_message = update.message.text
        history = context.user_data.setdefault("history", [])

        answer = await answer_question(state, user_id, user_message, history)

        # Обновляем память диалога, храним только последние N реплик
        history.append((user_message, answer))
        del history[: max(0, len(history) - cfg["history_max_turns"])]

        await reply_long_text(update, answer)

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    logger.info("Запускаю Telegram-бота (polling)", extra={"event": "bot_starting"})
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info(
        "Бот принимает сообщения; индексация материалов идёт в фоне",
        extra={"event": "bot_started"},
    )

    # Тяжёлая инициализация — в фоне, не блокирует приём /start
    asyncio.create_task(build_pipeline_background(cfg, state))

    try:
        await asyncio.Event().wait()
    finally:
        logger.info("Останавливаю бота", extra={"event": "bot_stopping"})
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Course Tutor QnA Bot — RAG-репетитор для Telegram")
    parser.add_argument(
        "--config",
        default="configs/settings.ini",
        help="Путь к .ini-файлу конфигурации (по умолчанию: configs/settings.ini)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["materials_path"], exist_ok=True)
    token = get_bot_token()
    asyncio.run(run_telegram_bot(cfg, token))


if __name__ == "__main__":
    main()
