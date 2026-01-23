"""Custom actions for the assistant."""
from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import EventType
import re
import requests

logger = logging.getLogger(__name__)


def _load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'").strip()
        if not key:
            continue
        existing = os.environ.get(key)
        if existing is None or existing in {"", "YOUR_NEW_KEY"}:
            os.environ[key] = value


_load_env_file(pathlib.Path(__file__).resolve().parents[1] / ".env")

OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/responses")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "20"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "300"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_ENABLE = os.getenv("OPENAI_ENABLE", "0") == "1"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models"
)
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "20"))

SYSTEM_PROMPT = os.getenv(
    "OPENAI_SYSTEM_PROMPT",
    """
Та бол Монгол хэлтэй эелдэг, товч, ойлгомжтой туслах чатбот.
- Хэрэглэгчийн асуултын утгыг ойлгож, утганд нь тохирсон хариу өг.
- Хэрвээ мэдээлэл дутуу бол 1 богино тодруулга асуу.
- Өгүүлбэртэй бодлого ирвэл шаардлагатай тооцоог хийж, богино тайлбар + эцсийн хариуг нэгжтэй нь өг.
- Худал мэдээлэл зохиож болохгүй. Мэдэхгүй бол үнэнээр нь хэл.
""".strip(),
)


def _collect_history(tracker: Tracker, max_messages: int = 8) -> List[Tuple[str, str]]:
    """Collect recent user/assistant messages in chronological order."""
    items: List[Tuple[str, str]] = []
    for event in reversed(tracker.events):
        event_type = event.get("event")
        if event_type == "user":
            text = event.get("text")
            if text:
                items.append(("user", text))
        elif event_type == "bot":
            text = event.get("text")
            if text:
                items.append(("assistant", text))
        if len(items) >= max_messages:
            break

    items.reverse()
    return items


def _build_input_items(history: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    input_items: List[Dict[str, Any]] = []
    for role, text in history:
        if role == "assistant":
            content_type = "output_text"
        else:
            content_type = "input_text"
        input_items.append(
            {
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": text}],
            }
        )
    return input_items


def _extract_output_text(data: Dict[str, Any]) -> Optional[str]:
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content.get("text")
    return None


def _build_gemini_contents(history: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for role, text in history:
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    return contents


def _extract_gemini_text(data: Dict[str, Any]) -> Optional[str]:
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [p.get("text") for p in parts if p.get("text")]
    if texts:
        return "".join(texts)
    return None


def _call_gemini(history: List[Tuple[str, str]]) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    payload = {"contents": _build_gemini_contents(history)}
    url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
    if resp.status_code >= 400:
        logger.error("Gemini error %s: %s", resp.status_code, resp.text)
        return None
    data = resp.json()
    return _extract_gemini_text(data)


def _call_openai(history: List[Tuple[str, str]]) -> Optional[str]:
    if not OPENAI_ENABLE or not OPENAI_API_KEY:
        return None
    payload = {
        "model": OPENAI_MODEL,
        "instructions": SYSTEM_PROMPT,
        "input": _build_input_items(history),
        "max_output_tokens": OPENAI_MAX_TOKENS,
        "temperature": OPENAI_TEMPERATURE,
        "store": False,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        OPENAI_API_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT
    )
    if resp.status_code >= 400:
        logger.error("OpenAI error %s: %s", resp.status_code, resp.text)
        return None
    data = resp.json()
    return _extract_output_text(data)


class ActionLLMResponse(Action):
    def name(self) -> str:
        return "action_llm_response"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[EventType]:
        # First, try free local word-problem solver
        text = tracker.latest_message.get("text", "")
        answer = _solve_simple_word_problem(text)
        if answer:
            dispatcher.utter_message(text=answer)
            return []

        history = _collect_history(tracker)
        if not history:
            dispatcher.utter_message(text="Та асуултаа дахин бичнэ үү.")
            return []

        try:
            answer = _call_gemini(history)
        except requests.RequestException as exc:
            logger.exception("Gemini request failed: %s", exc)
            answer = None

        if not answer:
            try:
                answer = _call_openai(history)
            except requests.RequestException as exc:
                logger.exception("OpenAI request failed: %s", exc)
                answer = None

        if not answer:
            dispatcher.utter_message(
                text="Уучлаарай, одоогоор хариу үүсгэж чадсангүй. Дараа дахин оролдоорой."
            )
            return []

        dispatcher.utter_message(text=answer)
        return []


_ADD_WORDS = ("дахиад", "нэм", "нийт", "нийлбэр", "нэмж", "болоод")
_SUB_WORDS = (
    "хас",
    "аваад",
    "өгөөд",
    "өгвөл",
    "үлд",
    "хасагд",
    "идсэн",
    "идэж",
    "зарцуулсан",
    "алдсан",
    "дууссан",
)
_MUL_WORDS = ("тус бүр", "бүр", "үрж", "хайрцаг", "өдөрт")
_DIV_WORDS = ("хуваа", "хуваавал", "тэнцүү", "тус тус", "хэдэн хэсэг")

_UNIT_WORDS = (
    "хуудас",
    "км",
    "кг",
    "төгрөг",
    "ширхэг",
    "литр",
    "цаг",
    "минут",
    "өдөр",
    "үзэг",
    "алим",
    "бөмбөг",
    "дэвтэр",
    "чихэр",
)


def _normalize_number(token: str) -> float:
    # Handle thousand separators and decimals
    if "," in token and "." in token:
        token = token.replace(",", "")
    elif "," in token:
        # If comma is used as thousand separator (e.g., 100,000)
        parts = token.split(",")
        if all(len(p) == 3 for p in parts[1:]) and len(parts[0]) <= 3:
            token = "".join(parts)
        else:
            token = token.replace(",", ".")
    return float(token)


def _extract_numbers(text: str) -> List[float]:
    numbers = re.findall(r"\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?", text)
    result = []
    for n in numbers:
        token = n.replace(" ", "")
        result.append(_normalize_number(token))
    return result


def _extract_percent(text: str) -> Optional[float]:
    match = re.search(r"(\d{1,3}(?:[.,]\d+)?)\s*%", text)
    if match:
        return _normalize_number(match.group(1))
    match = re.search(r"(\d{1,3}(?:[.,]\d+)?)\s*хувь", text)
    if match:
        return _normalize_number(match.group(1))
    return None


def _detect_unit(text: str) -> Optional[str]:
    if "₮" in text:
        return "₮"
    for unit in _UNIT_WORDS:
        if unit in text:
            return unit
    return None


def _solve_simple_word_problem(text: str) -> Optional[str]:
    lower = text.lower()
    nums = _extract_numbers(lower)
    if not nums:
        return None

    unit = _detect_unit(lower)
    percent = _extract_percent(lower)

    if percent is not None and len(nums) >= 1:
        base = nums[0]
        is_discount = any(word in lower for word in ("хямдар", "хөнгөл", "буур", "хас"))
        is_increase = any(word in lower for word in ("нэмэгд", "өс", "нэмэх"))
        if is_discount and not is_increase:
            result = base * (1 - percent / 100)
            result_str = f"{result:g}"
            return f"{base:g} × (1 − {percent:g}%) = {result_str} {unit or ''}".strip()
        if is_increase and not is_discount:
            result = base * (1 + percent / 100)
            result_str = f"{result:g}"
            return f"{base:g} × (1 + {percent:g}%) = {result_str} {unit or ''}".strip()

    # Division (e.g., "45 чихэрийг 5 хүүхэд тэнцүү хуваавал ...")
    if any(word in lower for word in _DIV_WORDS) and len(nums) >= 2:
        result = nums[0] / nums[1]
        result_str = f"{result:g}"
        return f"Тэнцүү хуваавал: {result_str} {unit or ''}".strip()

    idx = 0
    result = nums[0]
    explanation = []

    # Multiplication hint (e.g., "өдөрт 35", "тус бүр 8")
    if any(word in lower for word in _MUL_WORDS) and len(nums) >= 2:
        result = nums[0] * nums[1]
        explanation.append(f"{nums[0]:g} × {nums[1]:g} = {result:g}")
        idx = 2

    remaining = nums[idx:]

    # Addition / subtraction hints
    if remaining:
        if any(word in lower for word in _SUB_WORDS) and not any(
            word in lower for word in _ADD_WORDS
        ):
            sub_value = remaining[-1]
            explanation.append(f"{result:g} − {sub_value:g} = {result - sub_value:g}")
            result -= sub_value
        else:
            add_value = sum(remaining)
            explanation.append(f"{result:g} + {add_value:g} = {result + add_value:g}")
            result += add_value

    if not explanation:
        # Default: sum of all numbers
        result = sum(nums)
        explanation.append(" + ".join(f"{n:g}" for n in nums) + f" = {result:g}")

    result_str = f"{result:g}"
    if unit:
        return f"{'; '.join(explanation)}. Нийт: {result_str} {unit}"
    return f"{'; '.join(explanation)}. Нийт: {result_str}"


class ActionSolveWordProblem(Action):
    def name(self) -> str:
        return "action_solve_word_problem"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[EventType]:
        history = _collect_history(tracker)
        if not history:
            dispatcher.utter_message(text="Та асуултаа дахин бичнэ үү.")
            return []

        # Prefer API-based reasoning when available
        answer = None
        try:
            answer = _call_gemini(history)
        except requests.RequestException as exc:
            logger.exception("Gemini request failed: %s", exc)
            answer = None

        if not answer:
            text = tracker.latest_message.get("text", "")
            answer = _solve_simple_word_problem(text)

        if not answer:
            try:
                answer = _call_openai(history)
            except requests.RequestException as exc:
                logger.exception("OpenAI request failed: %s", exc)
                answer = None

        if not answer:
            dispatcher.utter_message(
                text="Тоонуудыг олж чадсангүй. Бодлогоо арай тодорхой бичээд өгнө үү."
            )
            return []

        dispatcher.utter_message(text=answer)
        return []
