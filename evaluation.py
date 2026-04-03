"""
evaluation.py
=====================
Архитектура — 1 оценочный узел (LangGraph):

    [source EMR text]  +  [generated summary]
                  │
                  ▼
         Node ─ EvaluatorNode
                  │
                  ▼
           EvalReport (JSON)
         5 критериев, 20 бинарных проверок, max = 20

Принцип оценки — ZERO-TOLERANCE:
  Каждая проверка получает 1 ТОЛЬКО если факт присутствует, абсолютно точен,
  не обобщён и не искажён. Частичное совпадение = 0.

Адаптация под свою задачу:
  1. Замените критерии и проверки на релевантные вашей доменной области.
  2. Скорректируйте системный промпт (_EVAL_SYS) под вашу метрику качества.
  3. При необходимости — добавьте или уберите критерии в EvalReport.
  4. Обновите CRITERION_CHECKS и CRITERION_MAX при изменении набора проверок.

Зависимости:
  outlines (1.2.12) llama-cpp-python langgraph pydantic
"""

import sys
import re
import time
import logging
import argparse
from pathlib import Path

import outlines
from llama_cpp import Llama
from langgraph.graph import END, START, StateGraph

from typing import Any, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# КОНФИГ
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "path/to/your_model.gguf"
MAX_CTX     = 34000
MAX_INPUT   = 27000   # лимит символов исходного ЭМК
MAX_SUMMARY = 3000    # лимит символов оцениваемого резюме

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("eval")


def load_model(verbose: bool = False) -> Any:
    log.info(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    model = outlines.from_llamacpp(
        Llama(
            model_path=MODEL_PATH,
            n_ctx=MAX_CTX,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=verbose,
            chat_format=None,
        ),
        chat_mode=False,  # промпт собираем вручную через _format_prompt
    )
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit: return text
    log.warning(f"{label} truncated: {len(text)} → {limit} chars")
    return text[:limit]

# =============================================================================
# GRAPH STATE — состояние пайплайна оценки
# Передаётся между узлами LangGraph.
# =============================================================================

class EvalState(TypedDict):
    source_text:  str           # исходный текст ЭМК
    summary_text: str           # оцениваемое резюме
    model:        Any           # объект Outlines-модели
    eval_report:  Optional["EvalReport"]  # заполняется оценочным узлом


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# =============================================================================

def _format_prompt(system: str, user: str) -> str:
    """
    Формирует промпт в формате GigaChat (<|role_sep|> / <|message_sep|>).
    Замените разделители под chat-template вашей модели.
    Для Llama 3: <|begin_of_text|><|start_header_id|>system<|end_header_id|> и т.д.
    """
    ROLE_SEP = "<|role_sep|>\n"
    MSG_SEP  = "<|message_sep|>\n\n"
    return (
        f"system{ROLE_SEP}{system}{MSG_SEP}"
        f"user{ROLE_SEP}{user}{MSG_SEP}"
        f"assistant{ROLE_SEP}"
    )

# Шаблон user-части для EvaluatorNode (подставляется в _format_prompt):
#
# user_payload = (
#     f"=== ИСХОДНЫЙ ТЕКСТ ЭМК ===\n{source_text}\n\n"
#     f"=== РЕЗЮМЕ ДЛЯ ОЦЕНКИ ===\n{summary_text}\n\n"
#     "Оцени резюме по всем критериям согласно сверхжёсткой инструкции (Zero-Tolerance). "
#     "Для каждого check укажи 0 или 1. Обоснуй нули в rationale. Выведи только JSON."
# )


def _log_sep(node: str, phase: str):
    log.info(f"{'─'*55}\n  NODE: {node} | {phase}\n{'─'*55}")


# =============================================================================
# SYSTEM PROMPT — LLM-as-judge (Zero-Tolerance Clinical Auditor)
# Определяет роль судьи и глобальные правила оценки.
# Адаптируйте под свою доменную область и метрику строгости.
# =============================================================================

EVAL_SYS = """\
Ты — строгий клинический аудитор. Твоя задача — безжалостно проверять LLM-summary
медицинских ЭМК для КТ органов брюшной полости на предмет малейших потерь данных.

ГЛОБАЛЬНОЕ ПРАВИЛО ОЦЕНКИ (ZERO-TOLERANCE):
Любой check получает 1 ТОЛЬКО если информация присутствует, абсолютно точна,
не искажена и не обобщена.
Любой check получает 0 (ШТРАФ), если зафиксировано хотя бы одно из нарушений:
1. OMISSION (Пропуск): факт есть в source, но отсутствует в summary.
2. GENERALIZATION (Обобщение): конкретный факт заменён общим (потеря специфичности).
3. DISTORTION (Искажение): неверные цифры, смена диагноза, галлюцинация.
4. UNCERTAINTY: написано "не указано", хотя в source эти данные ЕСТЬ.

ЗАПРЕЩАЕТСЯ:
— ставить 1 за "частичное совпадение" или "почти правильно".
— прощать неполноту, если пропущен клинически релевантный факт.

Если факт реально отсутствует в самом исходном ЭМК (source) —
только тогда ты ставишь 1 (пропусков со стороны summary нет).
Действуй максимально строго. Находи все omissions.
Минимальные описания — соблюдай лимиты слов генерации.
"""


# =============================================================================
# CRITERION 1 — SafetyAlertEval
# Checks: 5  |  Max score: 5
# Охватывает: аллергию на контраст, креатинин, СКФ, нефрологические
#             диагнозы, нефротоксичные препараты
# =============================================================================

class SafetyAlertEval(BaseModel):
    """КРИТЕРИЙ 1 — БЕЗОПАСНОСТЬ (CRITICAL ALERTS): 0–5."""

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
            "1 = Точное числовое значение креатинина из ЭМК (или чёткий статус его отсутствия) "
            "перенесено без искажений.\n"
            "0 = В ЭМК есть значение креатинина, но в summary оно отсутствует, "
            "заменено на обобщение ('в норме') или указано как 'не известно'."
        )
    )
    check_egfr: Literal[0, 1] = Field(
        description=(
            "1 = Точное числовое значение СКФ/eGFR из ЭМК присутствует в summary.\n"
            "0 = СКФ есть в ЭМК, но отсутствует в summary, искажено или обобщено."
        )
    )
    check_nephro_diseases: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ нефрологические диагнозы (ХБП, СД, единственная почка и др.) "
            "из ЭМК отражены в summary явно и полно.\n"
            "0 = Пропущен хотя бы один нефрологический диагноз из источника."
        )
    )
    check_nephro_drugs: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ потенциально нефротоксичные препараты и препараты риска "
            "(метформин, антикоагулянты и др.) из ЭМК явно перечислены в summary.\n"
            "0 = Пропущен хотя бы один релевантный препарат из ЭМК."
        )
    )
    rationale: str = Field(
        description=(
            "В 1 предложении кратко опиши пропуски по безопасности. "
            "Строгое ограничение НЕ БОЛЕЕ 40 слов."
        )
    )


# =============================================================================
# CRITERION 2 — CompletenessEval
# Checks: 4  |  Max score: 4
# Охватывает: жалобы, анамнез, онкологический статус, лаб/инструментальные данные
# =============================================================================

class CompletenessEval(BaseModel):
    """КРИТЕРИЙ 2 — ПОЛНОТА ОХВАТА: 0–4."""

    check_complaints: Literal[0, 1] = Field(
        description=(
            "1 = Отражены ВСЕ ключевые жалобы и показания к исследованию из ЭМК.\n"
            "0 = Пропущена хотя бы одна значимая жалоба или жалобы сведены "
            "к размытому обобщению ('боли в животе' вместо конкретной локализации)."
        )
    )
    check_anamnesis: Literal[0, 1] = Field(
        description=(
            "1 = Отражены ВСЕ ключевые диагнозы, операции и значимые состояния анамнеза.\n"
            "0 = Пропущен хотя бы один значимый диагноз или операция."
        )
    )
    check_oncology: Literal[0, 1] = Field(
        description=(
            "1 = Онкологический статус отражён с сохранением всех деталей "
            "(точная локализация, стадия, текущее лечение).\n"
            "0 = Онко-диагноз пропущен или описан обобщённо (потеряна стадия, тип, локализация)."
        )
    )
    check_lab_imaging: Literal[0, 1] = Field(
        description=(
            "1 = Отражены ВСЕ клинически значимые результаты лаборатории "
            "и инструментальных исследований.\n"
            "0 = Пропущен хотя бы один ключевой результат (отклонение от нормы) из источника."
        )
    )
    rationale: str = Field(
        description=(
            "В 1 предложении укажи, какие именно факты, детали или жалобы "
            "были пропущены или обобщены. НЕ БОЛЕЕ 40 слов."
        )
    )


# =============================================================================
# CRITERION 3 — AccuracyEval
# Checks: 4  |  Max score: 4
# Охватывает: числовые данные, диагнозы, галлюцинации, интерпретацию
# =============================================================================

class AccuracyEval(BaseModel):
    """КРИТЕРИЙ 3 — ФАКТИЧЕСКАЯ ТОЧНОСТЬ: 0–4."""

    check_numbers: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ числовые данные (размеры, дозы, показатели) в summary абсолютно точны.\n"
            "0 = Хотя бы одно число искажено, критично округлено или перепутаны единицы."
        )
    )
    check_diagnoses: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ диагнозы переданы без искажения смысла, статуса и тяжести.\n"
            "0 = Подмена диагноза, потеря стадии/тяжести, или 'подозрение' превращено "
            "в 'утверждение' (и наоборот)."
        )
    )
    check_hallucinations: Literal[0, 1] = Field(
        description=(
            "1 = НЕТ ни одного факта, которого не было бы в исходном ЭМК.\n"
            "0 = Присутствует хотя бы одна галлюцинация: выдуманный препарат, "
            "симптом, рекомендация или дата."
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
        description=(
            "В 1 предложении опиши найденные искажения, выдумки или ложные интерпретации. "
            "НЕ БОЛЕЕ 40 слов."
        )
    )


# =============================================================================
# CRITERION 4 — StructureEval
# Checks: 3  |  Max score: 3
# Охватывает: наличие разделов, лаконичность, достаточный объём
# =============================================================================

class StructureEval(BaseModel):
    """КРИТЕРИЙ 4 — СТРУКТУРА И ФОРМАТ: 0–3."""

    check_sections: Literal[0, 1] = Field(
        description=(
            "1 = Присутствуют ВСЕ необходимые смысловые блоки "
            "(безопасность, жалобы, анамнез, лаб/инструменты).\n"
            "0 = Отсутствует хотя бы один ключевой раздел."
        )
    )
    check_conciseness: Literal[0, 1] = Field(
        description=(
            "1 = Текст лаконичен, нет смысловых дублей и информационного шума.\n"
            "0 = Явный шум или повторение одних и тех же фактов в разных абзацах."
        )
    )
    check_volume: Literal[0, 1] = Field(
        description=(
            "1 = Объём достаточен для полного покрытия всех значимых фактов без потерь.\n"
            "0 = Текст избыточно сжат, что привело к потере клинически значимых данных."
        )
    )
    rationale: str = Field(
        description=(
            "В 1 предложении укажи структурные проблемы: отсутствие разделов, "
            "избыточность или чрезмерное сжатие. НЕ БОЛЕЕ 40 слов."
        )
    )


# =============================================================================
# CRITERION 5 — ClinicalRelevanceEval
# Checks: 4  |  Max score: 4
# Охватывает: срочность КТ, онко-нефрологический контекст,
#             информационный шум, actionable-рекомендации
# =============================================================================

class ClinicalRelevanceEval(BaseModel):
    """КРИТЕРИЙ 5 — КЛИНИЧЕСКАЯ ЗНАЧИМОСТЬ: 0–4."""

    check_ct_priority: Literal[0, 1] = Field(
        description=(
            "1 = Срочность (cito) и критические показания для КТ явно и точно перенесены.\n"
            "0 = Указание на срочность потеряно, не отражено или искажено."
        )
    )
    check_onco_nephro_context: Literal[0, 1] = Field(
        description=(
            "1 = Онкологический и нефрологический контекст отражён исчерпывающе полно.\n"
            "0 = Хотя бы один аспект контекста упрощён, пропущен или потерял специфичность."
        )
    )
    check_no_noise: Literal[0, 1] = Field(
        description=(
            "1 = В summary нет информации, абсолютно нерелевантной для проведения КТ ОБП.\n"
            "0 = Присутствует нерелевантный информационный блок "
            "(например, детальное описание приёма у стоматолога)."
        )
    )
    check_actionable: Literal[0, 1] = Field(
        description=(
            "1 = ВСЕ значимые для рентгенолога рекомендации из ЭМК сохранены.\n"
            "0 = Пропущена хотя бы одна actionable-рекомендация, важная для процедуры КТ."
        )
    )
    rationale: str = Field(
        description=(
            "В 1 предложении опиши потери контекста, срочности или пропущенные рекомендации. "
            "НЕ БОЛЕЕ 40 слов."
        )
    )


# =============================================================================
# TOP-LEVEL EVAL REPORT
# Агрегирует все 5 критериев + итоговый вердикт.
# Передаётся как output_type в outlines.Generator — единственный тип на весь узел.
# =============================================================================

class EvalReport(BaseModel):
    """
    Полный отчёт об оценке резюме.
    output_type для outlines.Generator в EvaluatorNode.
    5 критериев, 20 бинарных проверок, максимальный балл = 20.
    """
    safety_alerts:      SafetyAlertEval
    completeness:       CompletenessEval
    accuracy:           AccuracyEval
    structure:          StructureEval
    clinical_relevance: ClinicalRelevanceEval

    summary_notes: str = Field(
        description=(
            "В 1–2 предложениях опиши жёсткий вердикт: конкретные пропуски (omissions), "
            "искажения и обобщения. НЕ БОЛЕЕ 80 слов."
        )
    )


# =============================================================================
# СПРАВОЧНИК ПРОВЕРОК — используется для подсчёта баллов по критериям
# =============================================================================

# Ключи бинарных проверок для каждого критерия
CRITERION_CHECKS: dict = {
    "safety_alerts":      [
        "check_allergy",
        "check_creatinine",
        "check_egfr",
        "check_nephro_diseases",
        "check_nephro_drugs",
    ],
    "completeness":       [
        "check_complaints",
        "check_anamnesis",
        "check_oncology",
        "check_lab_imaging",
    ],
    "accuracy":           [
        "check_numbers",
        "check_diagnoses",
        "check_hallucinations",
        "check_interpretation",
    ],
    "structure":          [
        "check_sections",
        "check_conciseness",
        "check_volume",
    ],
    "clinical_relevance": [
        "check_ct_priority",
        "check_onco_nephro_context",
        "check_no_noise",
        "check_actionable",
    ],
}

# Максимальный балл по каждому критерию (вычисляется автоматически)
CRITERION_MAX: dict = {k: len(v) for k, v in CRITERION_CHECKS.items()}
# {"safety_alerts": 5, "completeness": 4, "accuracy": 4, "structure": 3, "clinical_relevance": 4}
# Итого максимум: 20



# ─────────────────────────────────────────────────────────────
# ПОДСЧЕТ БАЛЛОВ
# ─────────────────────────────────────────────────────────────

def _compute_scores(report: EvalReport) -> dict:
    scores = {
        key: sum(getattr(report, key).model_dump()[k] for k in checks)
        for key, checks in CRITERION_CHECKS.items()
    }
    scores["total"] = sum(scores.values())
    return scores


def _score_grade(total: int) -> str:
    if total >= 19: return "Отлично — идеальная точность"
    if total >= 16: return "Хорошо — незначительные потери специфичности"
    if total >= 12: return "Удовлетворительно — присутствуют критичные пропуски"
    return "Неудовлетворительно — опасная потеря данных"


# ─────────────────────────────────────────────────────────────
# УЗЕЛ ОЦЕНКИ
# ─────────────────────────────────────────────────────────────

def evaluator_node(state: EvalState) -> EvalState:
    _log_sep("Evaluator", "START")
    t0 = time.time()
    user_payload = (
        f"=== ИСХОДНЫЙ ТЕКСТ ЭМК ===\n{state['source_text']}\n\n"
        f"=== РЕЗЮМЕ ДЛЯ ОЦЕНКИ ===\n{state['summary_text']}\n\n"
        "Оцени резюме по всем критериям согласно сверхжёсткой инструкции (Zero-Tolerance). "
        "Для каждого check укажи 0 или 1. Обоснуй нули в rationale. Выведи только JSON."
    )
    prompt = _format_prompt(system=EVAL_SYS, user=user_payload)
    result: str = state["model"](prompt, output_type=EvalReport, max_tokens=3000)
    sc = _compute_scores(result)
    log.info(
        f"  [EvalNode] {time.time()-t0:.1f}s | total={sc['total']}/20 | "
        f"safety={sc['safety_alerts']}/5 | completeness={sc['completeness']}/4 | "
        f"accuracy={sc['accuracy']}/4 | structure={sc['structure']}/3 | relevance={sc['clinical_relevance']}/4"
    )
    _log_sep("Evaluator", "END")
    return {**state, "eval_report": result}


# ─────────────────────────────────────────────────────────────
# СБОРКА ГРАФА И ЗАПУСК
# ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(EvalState)
    g.add_node("evaluator", evaluator_node)
    g.add_edge(START, "evaluator")
    g.add_edge("evaluator", END)
    return g.compile()


def run_evaluation(source_text: str, summary_text: str, model: Any):
    source_text  = _truncate(source_text,  MAX_INPUT,   "SOURCE")
    summary_text = _truncate(summary_text, MAX_SUMMARY, "SUMMARY")
    log.info(f"{'='*55}\nEVAL START | source={len(source_text)} chars | summary={len(summary_text)} chars")
    graph   = build_graph()
    initial: EvalState = {
        "source_text":  source_text,
        "summary_text": summary_text,
        "model":        model,
        "eval_report":  None,
    }
    t0    = time.time()
    final = graph.invoke(initial)
    log.info(f"EVAL END | {time.time()-t0:.1f}s\n{'='*55}")
    return final, time.time() - t0


# ─────────────────────────────────────────────────────────────
# ФОРМАТИРОВАНИЕ ТЕКСТОВОГО ОТЧЁТА
# ─────────────────────────────────────────────────────────────

def _checks_block(label: str, checks: dict, score: int, max_score: int, rationale: str) -> str:
    s = "-" * 60
    lines = "\n".join(f"    {'V' if v == 1 else 'X'}  {k}: {v}" for k, v in checks.items())
    return f"{s}\n  {label}   [{score} / {max_score}]\n{s}\n{lines}\n\n  [RATIONALE]\n  {rationale}\n"


def _format_report_txt(
    report: EvalReport,
    source_path: Path, summary_path: Path,
    model_load_sec: float, inference_sec: float,
    source_chars: int, summary_chars: int,
    generated_at: str,
) -> str:
    S, s = "=" * 60, "-" * 60
    sa, co, ac, st, cr = (
        report.safety_alerts, report.completeness, report.accuracy,
        report.structure, report.clinical_relevance
    )
    sc = _compute_scores(report)
    return "\n".join([
        S, "  ОТЧЕТ ОБ ОЦЕНКЕ КАЧЕСТВА (STRICT AUDIT)", S,
        f"  Дата/время:      {generated_at}",
        f"  Исходный ЭМК:    {source_path.name}  ({source_chars} символов)",
        f"  Резюме:          {summary_path.name}  ({summary_chars} символов)",
        f"  Время оценки:    {model_load_sec + inference_sec:.1f} сек", S,
        "  -- ИТОГОВЫЕ БАЛЛЫ ------------------------------------------",
        f"  Безопасность:          {sc['safety_alerts']:>3} / {CRITERION_MAX['safety_alerts']}",
        f"  Полнота охвата:        {sc['completeness']:>3} / {CRITERION_MAX['completeness']}",
        f"  Фактическая точность:  {sc['accuracy']:>3} / {CRITERION_MAX['accuracy']}",
        f"  Структура и формат:    {sc['structure']:>3} / {CRITERION_MAX['structure']}",
        f"  Клиническая значимость:{sc['clinical_relevance']:>3} / {CRITERION_MAX['clinical_relevance']}",
        s, f"  ИТОГО:                 {sc['total']:>3} / 20",
        f"  Уровень:  {_score_grade(sc['total'])}", S,
        "  -- КРИТИЧЕСКИЕ ОШИБКИ И ПРОПУСКИ ---------------------------",
        f"  {report.summary_notes}", "", S,
        _checks_block("КРИТЕРИЙ 1 — БЕЗОПАСНОСТЬ",
            {k: getattr(sa, k) for k in CRITERION_CHECKS["safety_alerts"]},
            sc["safety_alerts"], CRITERION_MAX["safety_alerts"], sa.rationale),
        _checks_block("КРИТЕРИЙ 2 — ПОЛНОТА ОХВАТА",
            {k: getattr(co, k) for k in CRITERION_CHECKS["completeness"]},
            sc["completeness"], CRITERION_MAX["completeness"], co.rationale),
        _checks_block("КРИТЕРИЙ 3 — ФАКТИЧЕСКАЯ ТОЧНОСТЬ",
            {k: getattr(ac, k) for k in CRITERION_CHECKS["accuracy"]},
            sc["accuracy"], CRITERION_MAX["accuracy"], ac.rationale),
        _checks_block("КРИТЕРИЙ 4 — СТРУКТУРА И ФОРМАТ",
            {k: getattr(st, k) for k in CRITERION_CHECKS["structure"]},
            sc["structure"], CRITERION_MAX["structure"], st.rationale),
        _checks_block("КРИТЕРИЙ 5 — КЛИНИЧЕСКАЯ ЗНАЧИМОСТЬ",
            {k: getattr(cr, k) for k in CRITERION_CHECKS["clinical_relevance"]},
            sc["clinical_relevance"], CRITERION_MAX["clinical_relevance"], cr.rationale),
        S,
    ])


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMR Summary Quality Evaluator (Strict)")
    parser.add_argument("--source_file",  required=True, help="Путь к исходному ЭМК (.txt)")
    parser.add_argument("--summary_file", required=True, help="Путь к оцениваемому резюме (.txt)")
    parser.add_argument("--out",   default=None, help="Путь для сохранения отчёта (.txt)")
    parser.add_argument("--debug", action="store_true", help="Подробные логи llama.cpp")
    args = parser.parse_args()

    source_path, summary_path = Path(args.source_file), Path(args.summary_file)
    for p in (source_path, summary_path):
        if not p.exists():
            log.error(f"File not found: {p}"); sys.exit(1)

    t_load = time.time()
    model  = load_model(verbose=args.debug)
    model_load_sec = time.time() - t_load

    source_text  = source_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    result, inference_sec = run_evaluation(source_text, summary_text, model)

    out_path = (
        Path(args.out) if args.out
        else source_path.with_name(source_path.stem + "_" + summary_path.stem + "_eval.txt")
    )
    report_txt = _format_report_txt(
        report=result["eval_report"],
        source_path=source_path, summary_path=summary_path,
        model_load_sec=model_load_sec, inference_sec=inference_sec,
        source_chars=len(source_text), summary_chars=len(summary_text),
        generated_at=generated_at,
    )
    out_path.write_text(report_txt, encoding="utf-8")
    log.info(f"Eval report → {out_path}")


if __name__ == "__main__":
    main()