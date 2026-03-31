"""
schemas_evaluation.py
=====================
Pydantic schemas (Outlines-compatible) and system prompt for the LLM-as-judge
evaluation pipeline. Pipeline: evaluation_V5.py

Architecture: single evaluation node. Input is a (source EMR, summary) pair.
Output is a structured EvalReport object with 5 criteria, 20 binary checks,
and per-criterion rationale fields.

Scoring principle — Zero-Tolerance:
  A check receives 1 ONLY IF the information is present, numerically exact,
  not generalized, and not hallucinated. Partial match = 0.

No runtime logic is included here.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# SYSTEM PROMPT — LLM-as-judge (Zero-Tolerance Clinical Auditor)
# =============================================================================

_EVAL_SYS = """\
Ты — строгий клинический аудитор. Твоя задача — безжалостно проверять LLM-summary медицинских ЭМК для КТ ОБП на предмет малейших потерь данных.

ГЛОБАЛЬНОЕ ПРАВИЛО ОЦЕНКИ (ZERO-TOLERANCE):
Любой check получает 1 ТОЛЬКО если информация присутствует, абсолютно точна, не искажена и не обобщена.
Любой check получает 0 (ШТРАФ), если зафиксировано хотя бы одно из нарушений:
1. OMISSION (Пропуск): Факт есть в source, но отсутствует в summary.
2. GENERALIZATION (Обобщение): Конкретный факт заменен на общий (потеря специфичности, стадии, локализации).
3. DISTORTION (Искажение): Неверные цифры, смена диагноза, додумывание (галлюцинация).
4. UNCERTAINTY: Написано "не указано", хотя в source эти данные ЕСТЬ.

ЗАПРЕЩАЕТСЯ:
— Ставить 1 за "частичное совпадение" или "почти правильно".
— Прощать неполноту, если пропущен клинически релевантный факт.
Отсутствие факта = ОШИБКА (0).
Обобщение вместо конкретики = ОШИБКА (0).

Если факт реально отсутствует в самом исходном ЭМК (source) — только тогда ты ставишь 1 (так как пропусков со стороны summary нет).
Действуй максимально строго. Находи все omissions.
Минимальные описания - соблюдай лимиты слов генерации.
"""

# User payload template (constructed at runtime):
#
# user_payload = (
#     f"=== ИСХОДНЫЙ ТЕКСТ ЭМК ===\n{source_text}\n\n"
#     f"=== РЕЗЮМЕ ДЛЯ ОЦЕНКИ ===\n{summary_text}\n\n"
#     "Оцени резюме по всем критериям согласно сверхжесткой инструкции (Zero-Tolerance). "
#     "Для каждого check укажи 0 или 1. Обоснуй нули в rationale. Выведи только JSON."
# )


# =============================================================================
# CRITERION 1 — SafetyAlertEval
# Checks: 5  |  Max score: 5
# Covers: contrast allergy, creatinine, eGFR, nephro diagnoses, nephro drugs
# =============================================================================

class SafetyAlertEval(BaseModel):
    """КРИТЕРИЙ 1 — БЕЗОПАСНОСТЬ (CRITICAL ALERTS): 0-5."""

    check_allergy: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ аллергические реакции (на контраст/йод и другие значимые), "
            "упомянутые в ЭМК, явно и конкретно перечислены в summary.\n"
            "0 = хотя бы одна аллергия из ЭМК пропущена, либо конкретные аллергены "
            "заменены на обобщение ('отягощенный аллергоанамнез' без деталей)."
        )
    )
    check_creatinine: Literal[0, 1] = Field(
        description=(
            "1 = Точное числовое значение креатинина из ЭМК (или четкий статус его отсутствия) перенесено без искажений.\n"
            "0 = В ЭМК есть значение креатинина, но в summary оно отсутствует, "
            "или заменено на обобщение ('в норме'), или указано как 'не известно' при наличии в источнике."
        )
    )
    check_egfr: Literal[0, 1] = Field(
        description=(
            "1 = Точное числовое значение СКФ/eGFR из ЭМК присутствует в summary.\n"
            "0 = СКФ есть в ЭМК, но отсутствует в summary, искажено или заменено на обобщенную фразу."
        )
    )
    check_nephro_diseases: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ присутствующие в ЭМК нефрологические диагнозы (ХБП, СД, единственная почка и др.) "
            "отражены в summary явно и полно.\n"
            "0 = Пропущен хотя бы один нефрологический диагноз или состояние, присутствующее в источнике."
        )
    )
    check_nephro_drugs: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ потенциально нефротоксичные препараты или препараты риска (метформин, антикоагулянты и др.) "
            "из ЭМК явно перечислены в summary.\n"
            "0 = Пропущен хотя бы один релевантный препарат из ЭМК."
        )
    )
    rationale: str = Field(
        description="В 1 предложении кратко опиши пропуски по безопасности. Строгое ограничение НЕ БОЛЕЕ 40 слов."
    )


# =============================================================================
# CRITERION 2 — CompletenessEval
# Checks: 4  |  Max score: 4
# Covers: complaints, anamnesis, oncology, lab/imaging
# =============================================================================

class CompletenessEval(BaseModel):
    """КРИТЕРИЙ 2 — ПОЛНОТА ОХВАТА: 0-4."""

    check_complaints: Literal[0, 1] = Field(
        description=(
            "1 = Отражены ВСЕ ключевые клинические жалобы и показания к исследованию из ЭМК.\n"
            "0 = Пропущена хотя бы одна клинически значимая жалоба, или специфические жалобы "
            "сведены к размытому обобщению (например, 'боли в животе' вместо 'острые боли в правом подреберье')."
        )
    )
    check_anamnesis: Literal[0, 1] = Field(
        description=(
            "1 = Отражены ВСЕ ключевые диагнозы, операции и значимые состояния из анамнеза в ЭМК.\n"
            "0 = Пропущен хотя бы один значимый диагноз/операция. Частичное соответствие недопустимо."
        )
    )
    check_oncology: Literal[0, 1] = Field(
        description=(
            "1 = Онкологический статус отражен с сохранением всех деталей из ЭМК "
            "(точная локализация, стадия, текущее лечение).\n"
            "0 = Онко-диагноз пропущен ИЛИ описан обобщенно (потеряны стадия, тип или локализация), "
            "если эти детали были в оригинале."
        )
    )
    check_lab_imaging: Literal[0, 1] = Field(
        description=(
            "1 = Отражены ВСЕ клинически значимые (критические) результаты лаборатории и инструментальных исследований.\n"
            "0 = Пропущен хотя бы один ключевой результат (отклонение от нормы), указанный в источнике."
        )
    )
    rationale: str = Field(
        description="В 1 предложении кратко укажи, какие именно факты, детали или жалобы были пропущены или обобщены. Строгое ограничение НЕ БОЛЕЕ 40 слов."
    )


# =============================================================================
# CRITERION 3 — AccuracyEval
# Checks: 4  |  Max score: 4
# Covers: numeric values, diagnoses, hallucinations, interpretation
# =============================================================================

class AccuracyEval(BaseModel):
    """КРИТЕРИЙ 3 — ФАКТИЧЕСКАЯ ТОЧНОСТЬ: 0-4."""

    check_numbers: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ перенесенные в summary числовые данные (размеры, дозы, показатели) абсолютно точны.\n"
            "0 = Хотя бы одно число искажено, критично округлено или перепутаны единицы измерения."
        )
    )
    check_diagnoses: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ диагнозы переданы строго без искажения смысла, статуса и тяжести.\n"
            "0 = Зафиксирована подмена диагноза, потеря стадии/тяжести, или 'подозрение' превращено в 'утверждение' (и наоборот)."
        )
    )
    check_hallucinations: Literal[0, 1] = Field(
        description=(
            "1 = НЕТ ни одного факта, которого не было бы в исходном ЭМК.\n"
            "0 = Присутствует хотя бы одна галлюцинация: выдуманный препарат, симптом или рекомендация."
        )
    )
    check_interpretation: Literal[0, 1] = Field(
        description=(
            "1 = Интерпретация строго следует данным ЭМК, без отсебятины.\n"
            "0 = Summary содержит клинические выводы, которых нет в источнике, "
            "или присутствует субъективное усиление/смягчение тяжести состояния."
        )
    )
    rationale: str = Field(
        description="В 1 предложении кратко опиши найденные искажения, выдумки или ложные интерпретации. Строгое ограничение НЕ БОЛЕЕ 40 слов."
    )


# =============================================================================
# CRITERION 4 — StructureEval
# Checks: 3  |  Max score: 3
# Covers: required sections present, conciseness, sufficient volume
# =============================================================================

class StructureEval(BaseModel):
    """КРИТЕРИЙ 4 — СТРУКТУРА И ФОРМАТ: 0-3."""

    check_sections: Literal[0, 1] = Field(
        description=(
            "1 = Присутствуют ВСЕ необходимые смысловые блоки (безопасность, жалобы, анамнез, лаб/инструменты).\n"
            "0 = Отсутствует хотя бы один ключевой раздел."
        )
    )
    check_conciseness: Literal[0, 1] = Field(
        description=(
            "1 = Текст лаконичен, нет смысловых дублей и отвлекающего шума.\n"
            "0 = Присутствует явный шум, повторение одних и тех же фактов в разных абзацах."
        )
    )
    check_volume: Literal[0, 1] = Field(
        description=(
            "1 = Объем достаточен для полного покрытия всех значимых фактов исходника без их потери.\n"
            "0 = Текст слишком сжат, что привело к потере клинически значимых данных."
        )
    )
    rationale: str = Field(
        description="В 1 предложении кратко укажи структурные проблемы: отсутствие разделов, избыточность или чрезмерное сжатие. Строгое ограничение НЕ БОЛЕЕ 40 слов."
    )


# =============================================================================
# CRITERION 5 — ClinicalRelevanceEval
# Checks: 4  |  Max score: 4
# Covers: CT urgency, onco-nephro context, noise, actionable recommendations
# =============================================================================

class ClinicalRelevanceEval(BaseModel):
    """КРИТЕРИЙ 5 — КЛИНИЧЕСКАЯ ЗНАЧИМОСТЬ: 0-4."""

    check_ct_priority: Literal[0, 1] = Field(
        description=(
            "1 = Срочность (cito) и критические показания для КТ явно и точно перенесены из ЭМК.\n"
            "0 = Указание на срочность потеряно, не отражено или искажено."
        )
    )
    check_onco_nephro_context: Literal[0, 1] = Field(
        description=(
            "1 = Онкологический и нефрологический контекст отражен исчерпывающе полно.\n"
            "0 = Хотя бы один аспект контекста упрощен, пропущен или потерял специфичность."
        )
    )
    check_no_noise: Literal[0, 1] = Field(
        description=(
            "1 = В summary нет информации, абсолютно нерелевантной для проведения КТ ОБП.\n"
            "0 = Присутствует хотя бы один нерелевантный информационный блок (например, детальное описание приема у стоматолога)."
        )
    )
    check_actionable: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ значимые для рентгенолога рекомендации из ЭМК сохранены.\n"
            "0 = Пропущена хотя бы одна actionable рекомендация, важная для процедуры КТ."
        )
    )
    rationale: str = Field(
        description="В 1 предложении кратко опиши потери контекста, потерю срочности или пропущенные рекомендации. Строгое ограничение НЕ БОЛЕЕ 40 слов."
    )


# =============================================================================
# TOP-LEVEL EVAL REPORT
# Aggregates all 5 criteria + global summary verdict.
# This is the output_type passed to outlines.Generator.
# =============================================================================

class EvalReport(BaseModel):
    """
    Full evaluation report. Passed as output_type to outlines.Generator.
    5 criteria, 20 binary checks, max total score = 20.
    """
    safety_alerts:      SafetyAlertEval
    completeness:       CompletenessEval
    accuracy:           AccuracyEval
    structure:          StructureEval
    clinical_relevance: ClinicalRelevanceEval

    summary_notes: str = Field(
        description=(
            "В 1-2 предложений кратко опиши жесткий вердикт: перечислить конкретные пропуски (omissions), "
            "искажения и обобщения. Строгое ограничение НЕ БОЛЕЕ 80 слов."
        )
    )


# =============================================================================
# SCORING REFERENCE
# =============================================================================

# Check keys per criterion (for score computation):
CRITERION_CHECKS: dict = {
    "safety_alerts":      ["check_allergy", "check_creatinine", "check_egfr", "check_nephro_diseases", "check_nephro_drugs"],
    "completeness":       ["check_complaints", "check_anamnesis", "check_oncology", "check_lab_imaging"],
    "accuracy":           ["check_numbers", "check_diagnoses", "check_hallucinations", "check_interpretation"],
    "structure":          ["check_sections", "check_conciseness", "check_volume"],
    "clinical_relevance": ["check_ct_priority", "check_onco_nephro_context", "check_no_noise", "check_actionable"],
}

CRITERION_MAX: dict = {k: len(v) for k, v in CRITERION_CHECKS.items()}
# {"safety_alerts": 5, "completeness": 4, "accuracy": 4, "structure": 3, "clinical_relevance": 4}
# Total maximum: 20


# Score grade thresholds:
# >= 19 → Отлично — идеальная точность
# >= 16 → Хорошо — незначительные потери специфичности
# >= 12 → Удовлетворительно — присутствуют критичные пропуски
#  < 12 → Неудовлетворительно — опасная потеря данных
