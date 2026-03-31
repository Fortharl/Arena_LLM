"""
schemas_summary.py
==================
Pydantic schemas (Outlines-compatible) and system prompts for the EMR summarization pipeline.
Pipeline: graph_summary_V6.py

Architecture: 4 extraction nodes (structured JSON output via constrained decoding)
              + 1 composition node (free-text summary from aggregated JSON facts).

Each schema is passed directly to outlines.Generator(model, output_type=<Schema>),
which translates the Pydantic model into a finite automaton constraining token generation
at the logit level. Field(description=...) serves as both the JSON Schema annotation
and the per-field instruction to the model.

No runtime logic is included here. For the full pipeline, see graph_summary_V6.py.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# SHARED PRIMITIVES
# =============================================================================

class NumericValue(BaseModel):
    """A numeric clinical measurement with its unit and provenance quote."""
    value: Optional[float] = Field(
        description="Числовое значение показателя. null если число не извлечено."
    )
    unit: Optional[str] = Field(
        description="Единица измерения показателя (мкмоль/л, ммоль/л и т.д.)."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено значение."
    )


class RenalFunction(BaseModel):
    """Renal function indicators relevant to contrast administration safety."""
    creatinine: Optional[NumericValue] = Field(
        description="Последнее значение креатинина (число + единица)."
    )
    egfr: Optional[NumericValue] = Field(
        description="Последнее значение СКФ (eGFR)."
    )
    creatinine_date: Optional[str] = Field(
        description="Дата анализа креатинина. null если не указана."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


# =============================================================================
# NODE 1 — SafetyExtractor
# Schema: SafetyData
# Extracts: contrast allergy, renal function, nephrotoxic/risk medications
# =============================================================================

_SAFETY_SYS = """\
Ты — медицинский экстрактор.

Извлекай ТОЛЬКО факты.

ЗАПРЕЩЕНО:
— делать выводы
— интерпретировать значения
— вычислять стадии

ПРАВИЛА:
— если нет числа → value=null
— обязательно добавляй source_quote

ПРИМЕР:
"Креатинин 110 мкмоль/л"
→ value=110, unit="мкмоль/л"
"""

# User prompt template (formatted at runtime):
# f"Извлеки данные безопасности из ЭМК:\n\n{emr_text}"


class SafetyData(BaseModel):
    """
    Node 1 output schema.
    Covers all safety-critical data for contrast-enhanced CT:
    allergy status, renal function, and nephrotoxic/risk medications.
    """
    allergy_contrast: bool = Field(
        description="True если есть ЛЮБОЕ упоминание аллергии на контраст/йод/гадолиний."
    )
    allergy_details: str = Field(
        description="Короткая цитата с описанием аллергии."
    )
    renal: RenalFunction = Field(
        description="Данные о функции почек."
    )
    anticoagulants: List[str] = Field(
        description="Список антикоагулянтов/антиагрегантов из текста."
    )
    metformin: str = Field(
        description="Метформин или аналоги с дозой."
    )
    nephrotoxic_nsaids: List[str] = Field(
        description="НПВС с нефротоксичностью."
    )
    chemo_nephrotoxic: List[str] = Field(
        description="Нефротоксичные химиопрепараты."
    )
    diuretics: List[str] = Field(
        description="Диуретики."
    )


# =============================================================================
# NODE 2 — AnatomyClinicalExtractor
# Schema: AnatomyClinicalData (+ AbdominalOp sub-schema)
# Extracts: complaints, disease timeline, GI diagnoses, abdominal surgery history
# =============================================================================

_ANATOMY_SYS = """\
Ты — медицинский экстрактор.

Извлекай факты без интерпретации.

ПРИМЕР:
"желчный пузырь удалён"
→ cholecystectomy_done = true
"""

# User prompt template (formatted at runtime):
# f"Извлеки клинические и анатомические данные из ЭМК:\n\n{emr_text}"


class AbdominalOp(BaseModel):
    """A single prior abdominal surgical procedure."""
    operation: str = Field(
        description="Название операции на органах брюшной полости."
    )
    date_approx: Optional[str] = Field(
        description="Дата или год операции."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class AnatomyClinicalData(BaseModel):
    """
    Node 2 output schema.
    Covers clinical and anatomical context: complaints, disease timeline,
    relevant GI diagnoses, surgical history.
    """
    main_complaint: str = Field(
        description="Главные жалобы текущего обращения."
    )
    disease_history_timeline: str = Field(
        description="Краткая хронология заболевания."
    )
    trigger_factor: str = Field(
        description="Провоцирующий фактор."
    )
    critical_diseases: str = Field(
        description="Заболевания с риском для почек."
    )
    diabetes_type: str = Field(
        description="Тип сахарного диабета."
    )
    stool_delay_days: str = Field(
        description="Задержка стула."
    )
    preliminary_diagnosis: str = Field(
        description="Предварительный диагноз."
    )
    abdominal_ops: List[AbdominalOp] = Field(
        description="Операции на ОБП."
    )
    cholecystectomy_done: bool = Field(
        description="Удалён ли желчный пузырь."
    )
    stoma_or_ostomy: str = Field(
        description="Наличие стомы."
    )
    radiation_abd_pelvis: str = Field(
        description="Лучевая терапия."
    )
    gi_diseases: List[str] = Field(
        description="Заболевания ЖКТ."
    )
    smoking_history: str = Field(
        description="Курение."
    )
    family_history: str = Field(
        description="Семейный анамнез."
    )


# =============================================================================
# NODE 3a — OncoExtractor
# Schema: OncoData (+ OncologyRecord sub-schema)
# Extracts: all oncological diagnoses with TNM, treatment, response, remission
# =============================================================================

_ONCO_SYS = """\
Ты — экстрактор онкологии.

ЗАПРЕЩЕНО:
— объединять разные опухоли
— делать выводы

ПРАВИЛА:
— одна опухоль = одна запись
— извлекать ВСЕ случаи (даже старые)
— source_quote обязателен

ПРИМЕР:
"рак сигмовидной кишки 2014"
→ diagnosis_year=2014
"""

# User prompt template (formatted at runtime):
# f"Извлеки онкологический анамнез из ЭМК:\n\n{emr_text}"


class OncologyRecord(BaseModel):
    """A single oncological diagnosis with full staging and treatment context."""
    primary_site: str = Field(
        description="Локализация первичной опухоли."
    )
    stage_tnm: str = Field(
        description="Стадия TNM если указана."
    )
    diagnosis_year: Optional[int] = Field(
        description="Год постановки диагноза."
    )
    is_primary_abd: bool = Field(
        description="True если опухоль в ОБП/МТ."
    )
    metastatic_sites_abd: List[str] = Field(
        description="Метастазы в ОБП/МТ."
    )
    peritoneal_involvement: str = Field(
        description="Поражение брюшины."
    )
    current_treatment: str = Field(
        description="Текущее лечение."
    )
    treatment_response: str = Field(
        description="Ответ на лечение."
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
    Node 3a output schema.
    All oncological diagnoses as separate records; flag for prior abdominal CT.
    """
    oncology: List[OncologyRecord] = Field(
        description="ВСЕ онкологические диагнозы без ограничения количества."
    )
    ct_abd_previously: bool = Field(
        description="Была ли КТ ОБП ранее."
    )


# =============================================================================
# NODE 3b — LabImgExtractor
# Schema: LabImgData (+ LabRecord, ImagingFinding sub-schemas)
# Extracts: laboratory values with numeric precision; imaging findings
# =============================================================================

_LAB_IMG_SYS = """\
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

# User prompt template (formatted at runtime):
# f"Извлеки лабораторные данные и инструментальные находки из ЭМК:\n\n{emr_text}"


class LabRecord(BaseModel):
    """A single laboratory test result with value, unit and date."""
    test: str = Field(
        description="Название лабораторного показателя."
    )
    value: Optional[float] = Field(
        description="Числовое значение."
    )
    unit: Optional[str] = Field(
        description="Единица измерения."
    )
    date_approx: Optional[str] = Field(
        description="Дата анализа."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class ImagingFinding(BaseModel):
    """A single imaging study finding."""
    modality: str = Field(
        description="Тип исследования (КТ/УЗИ/МРТ)."
    )
    date_approx: Optional[str] = Field(
        description="Дата."
    )
    organ: str = Field(
        description="Орган."
    )
    finding: str = Field(
        description="Патологическая находка."
    )
    source_quote: str = Field(
        default="",
        description="Короткая точная цитата из текста, откуда извлечено."
    )


class LabImgData(BaseModel):
    """
    Node 3b output schema.
    Laboratory values and imaging findings extracted from the EMR.
    """
    labs: List[LabRecord] = Field(
        description="Лабораторные показатели."
    )
    imaging_abd: List[ImagingFinding] = Field(
        description="Инструментальные находки."
    )


# =============================================================================
# NODE 4 — SummaryComposer
# No output schema (free-text generation).
# Input: aggregated JSON dicts from Nodes 1-3b + regex-extracted demographics.
# =============================================================================

_SUMMARY_SYS = """\
Ты — врач-рентгенолог. Твоя задача — составить финальное клиническое резюме (Summary) на основе структурированных данных из ЭМК.

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
- КОНТРАСТ: (Если allergy_contrast=true: "АЛЛЕРГИЯ НА КОНТРАСТ: {allergy_details}". Иначе: "Противопоказаний не выявлено").
- ПОЧКИ: (Укажи есть ли заболевания почек. Укажи Креатинин и СКФ с числовыми значениями. Если creatinine_elevated=true: "ПОВЫШЕН КРЕАТИНИН").
- СОПУТСТВУЮЩИЕ ЗАБОЛЕВАНИЯ: (Заболевания, сопровождающиеся нарушением функции почек (сахарный диабет, подагра, артериальная гипертония, удаление одной почки, рак почки), а также гипертиреоз, тиреотоксикоз.")
- ПРЕПАРАТЫ: (Укажи Метформин и Антикоагулянты, если они есть. Если нет — перечисли кратко принимаемые препараты).

1. ПРИЧИНА ИССЛЕДОВАНИЯ И ЖАЛОБЫ:
- Сформулируй жалобы, ставшие причиной выполнения КТ органов брюшной полости.

2. АНАМНЕЗ ЗАБОЛЕВАНИЯ:
- Опиши развитие текущего состояния.

3. АНАМНЕЗ ЖИЗНИ И СОПУТСТВУЮЩАЯ ПАТОЛОГИЯ:
- Операции: Перечисли все.
- Хронические заболевания и Сопутствующие заболевания.
- Привычки пациента.

4. ЛАБОРАТОРНАЯ ДИАГНОСТИКА:
- Кратко перечисли значимые лабораторные данные (показатель, значение, дата).

5. ДАННЫЕ ИНСТРУМЕНТАЛЬНЫХ ОБСЛЕДОВАНИЙ:
- Резюмируй находки по инструментальным обследованиям.

Пиши лаконично, профессионально, без вводных фраз "В данном документе...". Начни сразу с CRITICAL ALERTS.

Максимум 800 слов. Начинай каждый раздел с его заголовка."""

# User payload template (constructed at runtime from aggregated node outputs):
#
# user_payload = (
#     f"=== ДАННЫЕ ПАЦИЕНТА (Python RegEx) ===\nПол: {sex_str}\nВозраст: {age_str}\n\n"
#     f"=== ФАКТЫ §Node1 ===\n{json.dumps(safety_data, ensure_ascii=False, indent=2)}\n\n"
#     f"=== ФАКТЫ §Node2 ===\n{json.dumps(anatomy_data, ensure_ascii=False, indent=2)}\n\n"
#     f"=== ФАКТЫ §Node3 (онкология) ===\n{json.dumps(onco_data, ensure_ascii=False, indent=2)}\n\n"
#     f"=== ФАКТЫ §Node4 (лаборатория и КТ) ===\n{json.dumps(lab_img_data, ensure_ascii=False, indent=2)}\n\n"
#     "Составь резюме строго по инструкции. Начни с §1 КОНТРАСТ."
# )
