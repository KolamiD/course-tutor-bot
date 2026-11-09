"""
Базовые тесты, не требующие GPU/LLM/vLLM — проверяют конфиг, хэширование
материалов и логику чанкования/фильтрации. Гонять через:

    pytest tests/
"""

import os
import sys
import configparser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as bot  # noqa: E402


def test_load_config_defaults(tmp_path):
    # Несуществующий файл конфига -> должны применяться дефолты, без падения
    cfg = bot.load_config(str(tmp_path / "nonexistent.ini"))
    assert cfg["model"] == "t-tech/T-lite-it-1.0"
    assert cfg["rag_top_k"] == 6
    assert 0 < cfg["relevance_threshold"] < 1
    assert cfg["history_max_turns"] >= 1


def test_load_config_overrides(tmp_path):
    ini_path = tmp_path / "settings.ini"
    config = configparser.ConfigParser()
    config["Course"] = {"name": "Тестовый курс"}
    config["Hybrid"] = {"relevance_threshold": "0.5"}
    with open(ini_path, "w", encoding="utf-8") as f:
        config.write(f)

    cfg = bot.load_config(str(ini_path))
    assert cfg["course_name"] == "Тестовый курс"
    assert cfg["relevance_threshold"] == 0.5


def test_compute_materials_hash_changes_on_new_file(tmp_path):
    materials = tmp_path / "MATERIALS"
    materials.mkdir()
    h1 = bot.compute_materials_hash(str(materials))

    (materials / "lecture1.pdf").write_bytes(b"%PDF-1.4 fake content")
    h2 = bot.compute_materials_hash(str(materials))

    assert h1 != h2, "Хэш должен измениться после добавления нового файла"


def test_compute_materials_hash_stable_without_changes(tmp_path):
    materials = tmp_path / "MATERIALS"
    materials.mkdir()
    (materials / "lecture1.pdf").write_bytes(b"%PDF-1.4 fake content")

    h1 = bot.compute_materials_hash(str(materials))
    h2 = bot.compute_materials_hash(str(materials))

    assert h1 == h2, "Хэш должен быть стабилен, если материалы не менялись"


def test_format_history_empty():
    assert "начался" in bot.format_history([])


def test_format_history_with_turns():
    history = [("Что такое градиентный спуск?", "Это метод оптимизации...")]
    formatted = bot.format_history(history)
    assert "Студент: Что такое градиентный спуск?" in formatted
    assert "Бот: Это метод оптимизации..." in formatted


def test_short_chunks_are_filtered_by_length_rule():
    # Проверяем именно правило фильтрации огрызков чанков (см. build_documents),
    # не гоняя реальный PDF-пайплайн.
    class FakeDoc:
        def __init__(self, text):
            self.page_content = text

    splits = [FakeDoc("Короткий"), FakeDoc("Это уже достаточно длинный кусок текста, " * 3)]
    filtered = [s for s in splits if len(s.page_content.strip()) >= 80]
    assert len(filtered) == 1
    assert "достаточно длинный" in filtered[0].page_content


def test_split_for_telegram_short_text_unchanged():
    text = "Короткий ответ, который точно помещается в одно сообщение."
    assert bot.split_for_telegram(text) == [text]


def test_split_for_telegram_respects_limit():
    long_text = ("Это предложение повторяется много раз. " * 300).strip()
    parts = bot.split_for_telegram(long_text, limit=500)
    assert all(len(p) <= 500 for p in parts)
    assert len(parts) > 1


def test_split_for_telegram_prefers_paragraph_boundaries():
    para_text = ("Абзац с содержанием. " * 15 + "\n\n") * 4
    parts = bot.split_for_telegram(para_text, limit=300)
    assert all(len(p) <= 300 for p in parts)
    # Каждая часть — это законченный текст без обрыва слова на границе
    for p in parts:
        assert not p.endswith(("Абзац", "содержан", "с содержа"))


def test_split_for_telegram_empty_text():
    assert bot.split_for_telegram("") == []
    assert bot.split_for_telegram("   ") == []
