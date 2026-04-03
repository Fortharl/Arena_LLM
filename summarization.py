"""
summarization.py
========================
Архитектура — 4 узла извлечения + 1 узел суммаризации (LangGraph):

    [EMR text]
        │
        ▼
    Node 1 ─ SafetyExtractor       → SafetyData
        │
        ▼
    Node 2 ─ AnatomyClinicalExtractor → AnatomyClinicalData
        │
        ▼
    Node 3 ─ OncoExtractor         → OncoData
        │
        ▼
    Node 4 ─ LabImgExtractor       → LabImgData
        │
        ▼
    Node 5 ─ SummaryComposer        → final_summary (plain text)

Каждый узел 1–4 использует направленную генерацию через Outlines
(outlines.Generator с output_type=<PydanticModel>). Узел 5 получает
агрегированный JSON от всех предыдущих узлов и генерирует свободный текст.

Принцип декомпозиции: извлечение фактов и синтез — разные когнитивные задачи.
На малых LLM их раздельное решение даёт лучший результат, чем один большой промпт.

Адаптация под свою задачу:
  1. Замените поля в схемах на нужные вам сущности.
  2. Скорректируйте системные промпты под вашу доменную область.
  3. Обновите _format_prompt() под chat-template вашей модели.
  4. В SummaryComposer замените структуру вывода под свой шаблон резюме.
"""

import sys
import re
import json
import time
import logging
import argparse
from pathlib import Path

import outlines
from llama_cpp import Llama
from langgraph.graph import END, START, StateGraph

from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# КОНФИГ — измените под свою модель и железо
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "path/to/your_model.gguf"
MAX_CTX    = 40000   # контекстное окно модели
MAX_INPUT  = 35000   # лимит символов входного ЭМК

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_run.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("emr")


def load_model(verbose: bool = False) -> Any:
    log.info(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    llm = outlines.models.LlamaCpp(
        Llama(
            model_path=MODEL_PATH,
            n_ctx=MAX_CTX,
            n_threads=8,
            n_gpu_layers=-1,  # -1 = всё на GPU; 0 = только CPU
            verbose=verbose,
        )
    )
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    return llm


def _truncate(text: str) -> str:
    """Обрезает входной текст до MAX_INPUT символов с предупреждением."""
    if len(text) <= MAX_INPUT:
        return text
    log.warning(f"EMR truncated: {len(text)} → {MAX_INPUT} chars")
    return text[:MAX_INPUT]


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# Замените _format_prompt под chat-template своей модели.
# Llama 3 / GigaChat имеют разные разделители — сверьтесь с документацией.
# =============================================================================

def _format_prompt(system: str, user: str) -> str:
    """
    Формирует промпт в формате конкретной модели.
    Этот шаблон — для Llama 3 (<|begin_of_text|> / <|start_header_id|>).
    Для GigaChat используйте: system<|role_sep|>...<|message_sep|>user<|role_sep|>...
    """
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_demographics(text: str):
    """
    RegEx-извлечение возраста и пола из заголовка ЭМК.
    Ищет паттерны вида 'Мужчина 64' / 'Женщина55'.
    Замени паттерн под формат своих документов.
    """
    head = text[:100]
    age, sex = "", ""
    match = re.search(r'(Мужчина|Женщина)\W*(\d{2})', head, re.IGNORECASE)
    if match:
        sex = "М" if match.group(1).lower().startswith("муж") else "Ж"
        age = match.group(2)
    return age, sex


# =============================================================================
# GRAPH STATE — общее состояние пайплайна LangGraph
# Каждый узел читает из state и возвращает {**state, новое_поле: значение}
# =============================================================================

class EMRState(TypedDict):
    emr_text:      str                        # исходный текст ЭМК
    model:         Any                        # объект llama_cpp модели
    generators:    Dict[str, Any]             # словарь outlines.Generator по узлам
    patient_age:   str                        # возраст, извлечённый через RegEx
    patient_sex:   str                        # пол, извлечённый через RegEx
    safety_data:   Optional["SafetyData"]
    anatomy_data:  Optional["AnatomyClinicalData"]
    onco_data:     Optional["OncoData"]
    lab_img_data:  Optional["LabImgData"]
    final_summary: Optional[str]              # финальный текст резюме


# =============================================================================
# NODE 1 — SafetyData
# Данные безопасности для КТ с контрастом:
# аллергии, функция почек, нефротоксичные препараты, антикоагулянты
# =============================================================================

# Системный промпт Node 1
SAFETY_SYS = """\
Ты — медицинский экстрактор.

Извлекай ТОЛЬКО факты.

ЗАПРЕЩЕНО:
— делать выводы
— интерпретировать значения
— вычислять стадии

ПРАВИЛА:
— если нет числа → value=null
— обязательно добавляй source_quote (короткая цитата из источника)

ПРИМЕР:
"Креатинин 110 мкмоль/л"
→ value=110, unit="мкмоль/л"
"""


class NumericValue(BaseModel):
    """Числовое значение с единицей измерения и цитатой из источника."""
    value: Optional[float] = Field(
        description="Числовое значение показателя. null если число не извлечено."
    )
    unit: Optional[str] = Field(
        description="Единица измерения (мкмоль/л, мл/мин/1.73м² и т.д.)."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено значение."
    )


class RenalFunction(BaseModel):
    """Функция почек: креатинин, СКФ, дата анализа."""
    creatinine: Optional[NumericValue] = Field(
        description="Последнее значение креатинина (число + единица)."
    )
    egfr: Optional[NumericValue] = Field(
        description="Последнее значение СКФ (eGFR), мл/мин/1.73м²."
    )
    creatinine_date: Optional[str] = Field(
        description="Дата анализа креатинина. null если не указана."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class SafetyData(BaseModel):
    """
    Выход Node 1. Критические данные безопасности для КТ с контрастом.
    Используется как output_type в outlines.Generator.
    """
    allergy_contrast: bool = Field(
        description="True если есть ЛЮБОЕ упоминание аллергии на контраст/йод/гадолиний."
    )
    allergy_details: str = Field(
        description="Короткая цитата с описанием аллергии (аллерген, реакция)."
    )
    renal: RenalFunction = Field(
        description="Данные о функции почек."
    )
    anticoagulants: List[str] = Field(
        description="Список антикоагулянтов/антиагрегантов из текста (варфарин, ЛМНГ и др.)."
    )
    metformin: str = Field(
        description="Метформин или аналоги (с дозой, если указана). Пустая строка если нет."
    )
    nephrotoxic_nsaids: List[str] = Field(
        description="НПВС с нефротоксичностью (диклофенак, ибупрофен и др.)."
    )
    chemo_nephrotoxic: List[str] = Field(
        description="Нефротоксичные химиопрепараты (цисплатин и др.)."
    )
    diuretics: List[str] = Field(
        description="Диуретики."
    )


# =============================================================================
# NODE 2 — AnatomyClinicalData
# Клинико-анатомический контекст: жалобы, анамнез, операции на ОБП
# =============================================================================

# Системный промпт Node 2
ANATOMY_SYS = """\
Ты — медицинский экстрактор.

Извлекай факты без интерпретации.

ПРИМЕР:
"желчный пузырь удалён"
→ cholecystectomy_done = true
"""


class AbdominalOp(BaseModel):
    """Одна операция на органах брюшной полости."""
    operation: str = Field(
        description="Название операции на органах брюшной полости."
    )
    date_approx: Optional[str] = Field(
        description="Дата или год операции. null если не указана."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class AnatomyClinicalData(BaseModel):
    """
    Выход Node 2. Клинические данные и хирургический анамнез.
    Используется как output_type в outlines.Generator.
    """
    main_complaint: str = Field(
        description="Главные жалобы текущего обращения."
    )
    disease_history_timeline: str = Field(
        description="Краткая хронология заболевания (когда началось, как развивалось)."
    )
    trigger_factor: str = Field(
        description="Провоцирующий фактор (травма, пища, стресс и т.д.). Пустая строка если нет."
    )
    critical_diseases: str = Field(
        description="Заболевания с риском для функции почек (СД, гипертония, подагра и т.д.)."
    )
    diabetes_type: str = Field(
        description="Тип сахарного диабета (1 / 2 / нет). Пустая строка если нет."
    )
    stool_delay_days: str = Field(
        description="Задержка стула в днях. Пустая строка если нет."
    )
    preliminary_diagnosis: str = Field(
        description="Предварительный диагноз из направления."
    )
    abdominal_ops: List[AbdominalOp] = Field(
        description="Все операции на органах брюшной полости и малого таза."
    )
    cholecystectomy_done: bool = Field(
        description="True если желчный пузырь удалён."
    )
    stoma_or_ostomy: str = Field(
        description="Наличие стомы (колостома, илеостома и т.д.). Пустая строка если нет."
    )
    radiation_abd_pelvis: str = Field(
        description="Лучевая терапия на ОБП/малый таз. Пустая строка если нет."
    )
    gi_diseases: List[str] = Field(
        description="Заболевания ЖКТ (болезнь Крона, ЯКБ и т.д.)."
    )
    smoking_history: str = Field(
        description="Курение (курит / бросил / нет)."
    )
    family_history: str = Field(
        description="Семейный анамнез (онкология и пр.). Пустая строка если нет."
    )


# =============================================================================
# NODE 3 — OncoData
# Онкологический анамнез: каждый диагноз как отдельная запись
# =============================================================================

# Системный промпт Node 3
ONCO_SYS = """\
Ты — экстрактор онкологии.

ЗАПРЕЩЕНО:
— объединять разные опухоли
— делать выводы

ПРАВИЛА:
— одна опухоль = одна запись в массиве oncology
— извлекать ВСЕ случаи (даже старые / в анамнезе)
— source_quote обязателен

ПРИМЕР:
"рак сигмовидной кишки 2014"
→ diagnosis_year=2014
"""


class OncologyRecord(BaseModel):
    """Один онкологический диагноз."""
    primary_site: str = Field(
        description="Локализация первичной опухоли (орган, сторона)."
    )
    stage_tnm: str = Field(
        description="Стадия TNM если указана. Пустая строка если нет."
    )
    diagnosis_year: Optional[int] = Field(
        description="Год постановки диагноза. null если не указан."
    )
    is_primary_abd: bool = Field(
        description="True если опухоль в брюшной полости или малом тазу."
    )
    metastatic_sites_abd: List[str] = Field(
        description="Метастазы в брюшной полости/малом тазу (список органов)."
    )
    peritoneal_involvement: str = Field(
        description="Поражение брюшины (канцероматоз и т.д.). Пустая строка если нет."
    )
    current_treatment: str = Field(
        description="Текущее лечение (химиотерапия, ЛТ, таргет, иммунотерапия)."
    )
    treatment_response: str = Field(
        description="Ответ на лечение из текста. Пустая строка если нет."
    )
    remission: bool = Field(
        description="True если указана ремиссия."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class OncoData(BaseModel):
    """
    Выход Node 3. Все онкологические диагнозы.
    Используется как output_type в outlines.Generator.
    """
    oncology: List[OncologyRecord] = Field(
        description="ВСЕ онкологические диагнозы без ограничения количества."
    )
    ct_abd_previously: bool = Field(
        description="True если КТ брюшной полости выполнялась ранее."
    )


# =============================================================================
# NODE 4 — LabImgData
# Лабораторные показатели и инструментальные находки
# =============================================================================

# Системный промпт Node 4
LAB_IMG_SYS = """\
Ты — экстрактор лабораторных данных.

ЗАПРЕЩЕНО:
— добавлять записи без числового значения
— копировать назначения

ПРАВИЛА:
— value = float
— source_quote обязателен

ПРИМЕР:
"АЛТ 45 Ед/л"
→ value=45
"""


class LabRecord(BaseModel):
    """Один лабораторный показатель с числовым значением."""
    test: str = Field(description="Название лабораторного показателя.")
    value: Optional[float] = Field(description="Числовое значение. null если нет.")
    unit: Optional[str] = Field(description="Единица измерения.")
    date_approx: Optional[str] = Field(description="Дата анализа. null если нет.")
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class ImagingFinding(BaseModel):
    """Одна инструментальная находка."""
    modality: str = Field(description="Тип исследования (КТ / УЗИ / МРТ / ПЭТ).")
    date_approx: Optional[str] = Field(description="Дата исследования. null если нет.")
    organ: str = Field(description="Орган или область.")
    finding: str = Field(description="Описание патологической находки.")
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class LabImgData(BaseModel):
    """
    Выход Node 4. Лабораторные и инструментальные данные.
    Используется как output_type в outlines.Generator.
    """
    labs: List[LabRecord] = Field(description="Лабораторные показатели.")
    imaging_abd: List[ImagingFinding] = Field(
        description="Находки инструментальных исследований (КТ, УЗИ, МРТ и пр.)."
    )


# =============================================================================
# NODE 5 — SummaryComposer (свободная генерация, не Pydantic)
# Получает на вход JSON-агрегат от Node 1–4, выдаёт текстовое резюме.
# Для этого узла output_type НЕ задаётся — используется outlines.Generator(model).
# =============================================================================

# Системный промпт Node 5 (шаблон резюме для КТ ОБП).
# Адаптируйте структуру под вашу задачу и формат документа.
SUMMARY_SYS = """\
Ты — врач-рентгенолог. Твоя задача — составить финальное клиническое резюме
на основе структурированных данных из ЭМК.

ЗАПРЕЩЕНО:
— интерпретировать значения
— делать новые диагнозы
— вычислять стадии

Используй ТОЛЬКО данные JSON.
Если данных нет → "не указано"

### СТРУКТУРА ВЫВОДА:

Возраст пациента: {age}
Пол пациента: {sex}

CRITICAL ALERTS:
- КОНТРАСТ: (если аллергия — указать аллерген и реакцию; иначе "Противопоказаний не выявлено")
- ПОЧКИ: (заболевания почек, Креатинин и СКФ с числовыми значениями)
- СОПУТСТВУЮЩИЕ ЗАБОЛЕВАНИЯ: (СД, гипертония, подагра и пр.)
- ПРЕПАРАТЫ: (метформин, антикоагулянты; если нет — краткий список препаратов)

1. ПРИЧИНА ИССЛЕДОВАНИЯ И ЖАЛОБЫ
2. АНАМНЕЗ ЗАБОЛЕВАНИЯ
3. АНАМНЕЗ ЖИЗНИ И СОПУТСТВУЮЩАЯ ПАТОЛОГИЯ
   - Операции, хронические заболевания, привычки
4. ЛАБОРАТОРНАЯ ДИАГНОСТИКА
   - Значимые показатели (название, значение, дата)
5. ДАННЫЕ ИНСТРУМЕНТАЛЬНЫХ ОБСЛЕДОВАНИЙ

Пиши лаконично, профессионально.
Максимум 800 слов. Начинай сразу с CRITICAL ALERTS.
"""

# Шаблон user-части для Node 5 (подставляется в _format_prompt как user):
#
# user_payload = (
#     f"=== ДАННЫЕ ПАЦИЕНТА (RegEx) ===\nПол: {sex}\nВозраст: {age}\n\n"
#     f"=== ФАКТЫ §Node1 (безопасность) ===\n{json.dumps(safety, ensure_ascii=False, indent=2)}\n\n"
#     f"=== ФАКТЫ §Node2 (клиника/анатомия) ===\n{json.dumps(anatomy, ensure_ascii=False, indent=2)}\n\n"
#     f"=== ФАКТЫ §Node3 (онкология) ===\n{json.dumps(onco, ensure_ascii=False, indent=2)}\n\n"
#     f"=== ФАКТЫ §Node4 (лаборатория/КТ) ===\n{json.dumps(lab_img, ensure_ascii=False, indent=2)}\n\n"
#     "Составь резюме строго по инструкции. Начни с CRITICAL ALERTS."
# )


# =============================================================================
# СПРАВОЧНИК УЗЛОВ — для быстрой ориентации
# =============================================================================

PIPELINE_NODES: dict = {
    "safety":   {"cls": "SafetyData",           "max_tokens": 1200, "node": "Node 1"},
    "anatomy":  {"cls": "AnatomyClinicalData",   "max_tokens": 1500, "node": "Node 2"},
    "onco":     {"cls": "OncoData",              "max_tokens": 1500, "node": "Node 3"},
    "lab_img":  {"cls": "LabImgData",            "max_tokens": 1500, "node": "Node 4"},
    "summary":  {"cls": None,                    "max_tokens": 1500, "node": "Node 5"},
    # summary: output_type=None → свободная генерация
}


# ─────────────────────────────────────────────────────────────
# УЗЛЫ ГРАФА
# ─────────────────────────────────────────────────────────────

def safety_extractor(state: EMRState) -> EMRState:
    _log_sep("SafetyExtractor", "START")
    t0 = time.time()
    prompt = _format_prompt(
        system=SAFETY_SYS,
        user=f"Извлеки данные безопасности из ЭМК:\n\n{state['emr_text']}"
    )
    result = state["generators"]["safety"](prompt, max_tokens=1200)
    log.info(f"  [Node1] {time.time()-t0:.1f}s | allergy={result.allergy_contrast} | metformin='{result.metformin}'")
    _log_sep("SafetyExtractor", "END")
    return {**state, "safety_data": result}


def anatomy_clinical_extractor(state: EMRState) -> EMRState:
    _log_sep("AnatomyClinicalExtractor", "START")
    t0 = time.time()
    prompt = _format_prompt(
        system=ANATOMY_SYS,
        user=f"Извлеки клинические и анатомические данные из ЭМК:\n\n{state['emr_text']}"
    )
    result = state["generators"]["anatomy"](prompt, max_tokens=1500)
    log.info(f"  [Node2] {time.time()-t0:.1f}s | ops={len(result.abdominal_ops)} | chol={result.cholecystectomy_done}")
    _log_sep("AnatomyClinicalExtractor", "END")
    return {**state, "anatomy_data": result}


def onco_extractor(state: EMRState) -> EMRState:
    _log_sep("OncoExtractor", "START")
    t0 = time.time()
    prompt = _format_prompt(
        system=ONCO_SYS,
        user=f"Извлеки онкологический анамнез из ЭМК:\n\n{state['emr_text']}"
    )
    result = state["generators"]["onco"](prompt, max_tokens=1500)
    log.info(f"  [Node3a] {time.time()-t0:.1f}s | onco={len(result.oncology)} | ct_prev={result.ct_abd_previously}")
    _log_sep("OncoExtractor", "END")
    return {**state, "onco_data": result}


def lab_img_extractor(state: EMRState) -> EMRState:
    _log_sep("LabImgExtractor", "START")
    t0 = time.time()
    prompt = _format_prompt(
        system=LAB_IMG_SYS,
        user=f"Извлеки лабораторные данные и инструментальные находки из ЭМК:\n\n{state['emr_text']}"
    )
    result = state["generators"]["lab_img"](prompt, max_tokens=2000)
    log.info(f"  [Node3b] {time.time()-t0:.1f}s | labs={len(result.labs)} | imaging={len(result.imaging_abd)}")
    _log_sep("LabImgExtractor", "END")
    return {**state, "lab_img_data": result}


def summary_composer(state: EMRState) -> EMRState:
    _log_sep("SummaryComposer", "START")
    t0 = time.time()
    s  = state["safety_data"].model_dump()  if state.get("safety_data")  else {}
    a  = state["anatomy_data"].model_dump() if state.get("anatomy_data") else {}
    o  = state["onco_data"].model_dump()    if state.get("onco_data")    else {}
    li = state["lab_img_data"].model_dump() if state.get("lab_img_data") else {}
    user_payload = (
        f"=== ДАННЫЕ ПАЦИЕНТА (RegEx) ===\n"
        f"Пол: {state['patient_sex'] or 'Не указан'}\n"
        f"Возраст: {state['patient_age'] or 'Не указан'}\n\n"
        f"=== ФАКТЫ §Node1 (безопасность) ===\n{json.dumps(s,  ensure_ascii=False, indent=2)}\n\n"
        f"=== ФАКТЫ §Node2 (клиника/анатомия) ===\n{json.dumps(a,  ensure_ascii=False, indent=2)}\n\n"
        f"=== ФАКТЫ §Node3 (онкология) ===\n{json.dumps(o,  ensure_ascii=False, indent=2)}\n\n"
        f"=== ФАКТЫ §Node4 (лаборатория/КТ) ===\n{json.dumps(li, ensure_ascii=False, indent=2)}\n\n"
        "Составь резюме строго по инструкции. Начни с CRITICAL ALERTS."
    )
    prompt = _format_prompt(system=SUMMARY_SYS, user=user_payload)
    try:
        raw_summary: str = state["generators"]["summary"](prompt, max_tokens=1500)
    except Exception as e:
        log.error(f"[Node4] FAILED: {e}")
        raw_summary = "[ОШИБКА ГЕНЕРАЦИИ РЕЗЮМЕ]"
    log.info(f"  [Node4] {time.time()-t0:.1f}s | chars={len(raw_summary)}")
    _log_sep("SummaryComposer", "END")
    return {**state, "final_summary": raw_summary}

def _log_sep(node: str, phase: str):
    log.info(f"{'─'*55}\n  NODE: {node} | {phase}\n{'─'*55}")

# ─────────────────────────────────────────────────────────────
# СБОРКА ГРАФА И ЗАПУСК
# ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(EMRState)
    g.add_node("safety",  safety_extractor)
    g.add_node("anatomy", anatomy_clinical_extractor)
    g.add_node("onco",    onco_extractor)
    g.add_node("lab_img", lab_img_extractor)
    g.add_node("summary", summary_composer)
    g.add_edge(START,      "safety")
    g.add_edge("safety",   "anatomy")
    g.add_edge("anatomy",  "onco")
    g.add_edge("onco",     "lab_img")
    g.add_edge("lab_img",  "summary")
    g.add_edge("summary",  END)
    return g.compile()


def run_pipeline(emr_text: str, model: Any) -> EMRState:
    emr_text   = _truncate(emr_text)
    age, sex   = extract_demographics(emr_text)
    log.info(f"{'═'*55}\nPIPELINE START | EMR: {len(emr_text)} chars | sex={sex} age={age}")
    generators = {
        "safety":  outlines.Generator(model, output_type=SafetyData),
        "anatomy": outlines.Generator(model, output_type=AnatomyClinicalData),
        "onco":    outlines.Generator(model, output_type=OncoData),
        "lab_img": outlines.Generator(model, output_type=LabImgData),
        "summary": outlines.Generator(model),  # свободная генерация для текста резюме
    }
    graph = build_graph()
    initial: EMRState = {
        "emr_text":      emr_text,
        "model":         model,
        "generators":    generators,
        "patient_age":   age,
        "patient_sex":   sex,
        "safety_data":   None,
        "anatomy_data":  None,
        "onco_data":     None,
        "lab_img_data":  None,
        "final_summary": None,
    }
    t0    = time.time()
    final = graph.invoke(initial)
    log.info(f"PIPELINE END | {time.time()-t0:.1f}s total\n{'═'*55}")
    return final


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMR → CT Abdomen Summary")
    parser.add_argument("emr_file",        help="Путь к исходному ЭМК (.txt)")
    parser.add_argument("--out",   default=None, help="Путь для сохранения резюме (.txt)")
    parser.add_argument("--debug", action="store_true", help="Подробные логи llama.cpp")
    args = parser.parse_args()

    emr_path = Path(args.emr_file)
    if not emr_path.exists():
        log.error(f"File not found: {emr_path}"); sys.exit(1)

    model    = load_model(verbose=args.debug)
    emr_text = emr_path.read_text(encoding="utf-8")
    result   = run_pipeline(emr_text, model)
    summary  = result.get("final_summary") or "ERROR: summary not generated"

    out_path = Path(args.out) if args.out else emr_path.with_name(emr_path.stem + "_summary.txt")
    out_path.write_text(summary, encoding="utf-8")
    log.info(f"Summary → {out_path}")

    # Debug JSON — все извлечённые структурированные данные
    debug_path = emr_path.with_name(emr_path.stem + "_debug.json")
    debug_payload = {
        "demographics": {"age": result["patient_age"], "sex": result["patient_sex"]},
        "safety":   result["safety_data"].model_dump()   if result.get("safety_data")   else None,
        "anatomy":  result["anatomy_data"].model_dump()  if result.get("anatomy_data")  else None,
        "onco":     result["onco_data"].model_dump()     if result.get("onco_data")     else None,
        "lab_img":  result["lab_img_data"].model_dump()  if result.get("lab_img_data")  else None,
    }
    debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Debug JSON → {debug_path}")

    print(f"\n{'═'*55}\nИТОГОВОЕ РЕЗЮМЕ:\n{'═'*55}\n{summary}\n{'═'*55}")


if __name__ == "__main__":
    main()