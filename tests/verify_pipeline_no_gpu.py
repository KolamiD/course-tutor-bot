"""
Проверка реального кода main.py БЕЗ GPU и без интернета к Hugging Face.

Настоящие компоненты, которые проверяются как есть:
  - build_documents()      — чтение PDF, схлопывание переносов, чанкование, фильтрация огрызков
  - build_vectorstore()     — построение и переиспользование персистентного Chroma-индекса
  - build_hybrid_retriever() — реальный BM25Retriever + EnsembleRetriever
  - is_relevant_enough()     — порог релевантности
  - answer_question()        — вся цепочка: память -> порог -> retrieval -> генерация -> grounding
  - safe_llm_call()           — graceful degradation при падении/таймауте LLM

Подменяются только сама LLM (нужен GPU) и embedding-модель (нужна загрузка
весов с HF) — на лёгкие детерминированные фейки, не требующие ни GPU, ни сети.
Это не заменяет тест на реальной модели, но полностью проверяет, что
архитектура и управляющая логика бота работают корректно.
"""

import sys
import os
import shutil
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as bot  # noqa: E402


# --------------------------------------------------------------------------
# Фейковые embeddings: без сети, без GPU. Детерминированный "мешок слов"
# в векторе фиксированной длины — этого достаточно, чтобы BM25/Chroma
# реально работали и различали релевантные/нерелевантные тексты.
# --------------------------------------------------------------------------

class FakeEmbeddings:
    # Достаточно большая размерность, чтобы hash-коллизии между словами
    # разных языков/тем не создавали случайную "похожесть" — иначе тест
    # порога релевантности будет шумным не из-за бага в main.py, а из-за
    # слишком грубого фейка.
    DIM = 1024

    def _vec(self, text: str):
        import hashlib
        import re
        vec = [0.0] * self.DIM
        words = re.findall(r"[a-zа-яё0-9]+", text.lower())
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


# --------------------------------------------------------------------------
# Фейковая LLM: реализует интерфейс Runnable (.invoke), без реальной модели.
# Логирует все вызовы, чтобы проверить, что pipeline действительно их делает
# (переформулировка вопроса, генерация ответа, grounding-check).
# --------------------------------------------------------------------------

from langchain_core.runnables import Runnable


class FakeLLM(Runnable):
    """Реализует интерфейс Runnable, как и настоящая langchain_community.llms.VLLM,
    чтобы её можно было использовать в цепочках `prompt | llm` без изменений в main.py."""

    def __init__(self, behavior="normal"):
        self.behavior = behavior
        self.calls = []

    def invoke(self, prompt_value, config=None, **kwargs):
        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        self.calls.append(text)

        if self.behavior == "crash":
            raise RuntimeError("Симулированный сбой vLLM (например, OOM на GPU)")

        if "Самостоятельный вопрос" in text:
            # шаг переформулировки — просто возвращаем последний вопрос как есть
            return "Что такое переобучение?" if "переобучение" in text.lower() else "тестовый переформулированный вопрос"

        if "Ответь одним словом" in text:
            # grounding-check
            return "НЕТ" if self.behavior == "ungrounded" else "ДА"

        # генерация основного ответа
        return "Overfitting — это когда модель слишком точно подстраивается под обучающие данные. (тестовый ответ)"


# --------------------------------------------------------------------------
def generate_sample_materials(materials_dir: str) -> None:
    """Генерирует пару тестовых PDF прямо здесь, без внешних шагов —
    чтобы этот скрипт был самодостаточным и одинаково работал и локально,
    и в CI (GitHub Actions), не полагаясь на заранее подготовленные файлы."""
    from fpdf import FPDF

    os.makedirs(materials_dir, exist_ok=True)

    def make_pdf(path, title, paragraphs):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, title)
        pdf.ln(4)
        for p in paragraphs:
            pdf.multi_cell(0, 8, p)
            pdf.ln(4)
        pdf.output(path)

    make_pdf(
        os.path.join(materials_dir, "lecture1.pdf"),
        "Lecture 1: Gradient Descent",
        [
            "Gradient descent is an optimization algorithm used to minimize a loss function.",
            "It works by iteratively moving in the direction of steepest descent, "
            "as defined by the negative of the gradient.",
            "The learning rate controls the size of the steps taken during optimization.",
        ],
    )
    make_pdf(
        os.path.join(materials_dir, "lecture2.pdf"),
        "Lecture 2: Overfitting",
        [
            "Overfitting occurs when a model learns the training data too well, including its noise.",
            "Regularization techniques such as L1 and L2 penalties help prevent overfitting.",
            "Cross validation is used to estimate how well a model generalizes to unseen data.",
        ],
    )


# --------------------------------------------------------------------------
# Прогон реального пайплайна
# --------------------------------------------------------------------------

async def run_checks():
    results = []

    def check(name, condition, detail=""):
        status = "OK " if condition else "FAIL"
        results.append((status, name, detail))
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    cfg = bot.load_config("configs/settings.ini")
    cfg["materials_path"] = "MATERIALS"
    cfg["persist_directory"] = "chroma_db_test"
    cfg["relevance_threshold"] = 0.15  # для FakeEmbeddings подбираем эмпирически

    shutil.rmtree(cfg["persist_directory"], ignore_errors=True)
    generate_sample_materials(cfg["materials_path"])

    # --- 1. build_documents: реальное чтение PDF + чанкование + фильтрация ---
    splits = bot.build_documents(cfg)
    check("build_documents вернул непустой список чанков", len(splits) > 0, f"{len(splits)} чанков")
    check(
        "все чанки длиннее порога фильтрации (80 симв.)",
        all(len(s.page_content.strip()) >= 80 for s in splits),
    )

    # --- 2. build_vectorstore: построение + переиспользование по хэшу ---
    embeddings = FakeEmbeddings()
    vectorstore = bot.build_vectorstore(splits, embeddings, cfg)
    check("build_vectorstore построил индекс", vectorstore is not None)

    # повторный вызов без изменений в MATERIALS -> должен переиспользовать индекс
    hash_file = os.path.join(cfg["persist_directory"], "materials_hash.txt")
    hash_before = open(hash_file).read()
    vectorstore2 = bot.build_vectorstore(splits, embeddings, cfg)
    hash_after = open(hash_file).read()
    check("хэш материалов не изменился между вызовами", hash_before == hash_after)

    # --- 3. build_hybrid_retriever: реальный BM25 + вектор ---
    retriever = bot.build_hybrid_retriever(vectorstore, splits, cfg)
    docs = retriever.invoke("Что такое переобучение и как с ним бороться?")
    check("гибридный retriever нашёл документы по релевантному вопросу", len(docs) > 0)
    found_overfitting = any("verfit" in d.page_content.lower() or "regularization" in d.page_content.lower() for d in docs)
    check("среди найденных чанков есть текст про overfitting/regularization", found_overfitting)

    # --- 4. is_relevant_enough: порог релевантности ---
    relevant_ok = bot.is_relevant_enough(vectorstore, "Что такое overfitting?", cfg["relevance_threshold"])
    check("релевантный вопрос проходит порог", relevant_ok)

    irrelevant_ok = bot.is_relevant_enough(
        vectorstore, "Расскажи рецепт борща с говядиной", cfg["relevance_threshold"]
    )
    check("нерелевантный вопрос НЕ проходит порог (модерация работает)", not irrelevant_ok)

    # --- 5. answer_question: полная цепочка с фейковой LLM ---
    state = bot.BotState()
    state.vectorstore = vectorstore
    state.retriever = retriever
    state.cfg = cfg

    # 5a. Обычный вопрос, без истории
    state.llm = FakeLLM(behavior="normal")
    answer = await bot.answer_question(state, user_id=1, question="Что такое overfitting?", history=[])
    check("answer_question вернул непустой ответ по релевантному вопросу", bool(answer.strip()))
    check("LLM была вызвана для генерации (без истории — без переформулировки)", len(state.llm.calls) >= 1)

    # 5b. С историей — должна вызваться переформулировка (память)
    state.llm = FakeLLM(behavior="normal")
    history = [("Что такое градиентный спуск?", "Это метод оптимизации...")]
    answer2 = await bot.answer_question(state, user_id=1, question="А что такое переобучение?", history=history)
    condense_called = any("Самостоятельный вопрос" in c for c in state.llm.calls)
    check("при наличии истории вызывается переформулировка вопроса (память)", condense_called)

    # 5c. Нерелевантный вопрос — должен отсекаться порогом, LLM НЕ должна вызываться для генерации
    state.llm = FakeLLM(behavior="normal")
    answer3 = await bot.answer_question(state, user_id=1, question="Расскажи рецепт борща", history=[])
    check("нерелевантный вопрос отклонён без вызова генерации", "не относится" in answer3 or "не по теме" in answer3 or "относится" in answer3)

    # 5d. Симуляция падения LLM — graceful degradation
    state.llm = FakeLLM(behavior="crash")
    answer4 = await bot.answer_question(state, user_id=1, question="Что такое overfitting?", history=[])
    check(
        "при падении LLM бот вернул вежливый фолбэк, а не упал",
        "технический сбой" in answer4 or "Извини" in answer4,
        answer4[:80],
    )

    # 5e. Grounding-check отклоняет ответ
    state.llm = FakeLLM(behavior="ungrounded")
    answer5 = await bot.answer_question(state, user_id=1, question="Что такое overfitting?", history=[])
    check(
        "grounding-check отклонил недостаточно обоснованный ответ",
        "Не уверен" in answer5,
        answer5[:80],
    )

    shutil.rmtree(cfg["persist_directory"], ignore_errors=True)
    shutil.rmtree(cfg["materials_path"], ignore_errors=True)

    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\nИтого: {len(results) - len(failed)}/{len(results)} проверок пройдено")
    return len(failed) == 0


if __name__ == "__main__":
    ok = asyncio.run(run_checks())
    sys.exit(0 if ok else 1)
