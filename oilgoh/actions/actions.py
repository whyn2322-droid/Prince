from __future__ import annotations

import re
from typing import Dict, List, Optional, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

NUMBER_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)?")
EXPRESSION_PATTERN = re.compile(
    r"(-?\d+(?:[.,]\d+)?)(?:\s*)([+\-*/xX]|х)(?:\s*)(-?\d+(?:[.,]\d+)?)"
)
WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

ADD_KEYWORDS = [
    "нэм",
    "нэмэх",
    "нийл",
    "нийлбэр",
    "нийлүүлэх",
    "plus",
    "add",
    "нийт",
    "хамтдаа",
    "илүү",
]
SUB_KEYWORDS = [
    "хас",
    "хасах",
    "ялгавар",
    "ялгаа",
    "minus",
    "subtract",
    "авах",
    "авсан",
    "алдсан",
    "үлд",
    "үлдсэн",
    "хорогд",
    "дутаг",
    "дутуу",
    "бага",
]
MUL_KEYWORDS = [
    "үрж",
    "үржүүлэх",
    "үржвэр",
    "multiply",
    "times",
    "x",
]
DIV_KEYWORDS = [
    "хуваа",
    "хуваах",
    "хуваавал",
    "division",
    "divide",
    "ногдох",
    "per",
]


def extract_numbers(text: Text) -> List[float]:
    numbers: List[float] = []
    for match in NUMBER_PATTERN.findall(text):
        try:
            numbers.append(float(match.replace(",", ".")))
        except ValueError:
            continue
    return numbers


def format_number(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def parse_expression(text: Text) -> Optional[Dict[str, float]]:
    match = EXPRESSION_PATTERN.search(text)
    if not match:
        return None
    left = float(match.group(1).replace(",", "."))
    op_token = match.group(2)
    right = float(match.group(3).replace(",", "."))
    op_map = {"+": "add", "-": "sub", "*": "mul", "/": "div", "x": "mul", "X": "mul", "х": "mul"}
    return {"left": left, "right": right, "op": op_map[op_token], "symbol": op_token}


def tokenize(text: Text) -> List[Text]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def token_matches_keyword(token: Text, keyword: Text) -> bool:
    if token == keyword:
        return True
    return token.startswith(keyword)


def find_first_keyword(tokens: List[Text], keywords: List[str]) -> Optional[int]:
    for index, token in enumerate(tokens):
        for keyword in keywords:
            if token_matches_keyword(token, keyword):
                return index
    return None


def detect_operation(text: Text) -> Optional[str]:
    tokens = tokenize(text)
    if not tokens:
        return None
    candidates: Dict[str, int] = {}
    for op, keywords in [
        ("add", ADD_KEYWORDS),
        ("sub", SUB_KEYWORDS),
        ("mul", MUL_KEYWORDS),
        ("div", DIV_KEYWORDS),
    ]:
        idx = find_first_keyword(tokens, keywords)
        if idx is not None:
            candidates[op] = idx
    if not candidates:
        return None
    return min(candidates.items(), key=lambda item: item[1])[0]


def calculate(op: str, numbers: List[float]) -> Optional[float]:
    if not numbers:
        return None
    if op == "add":
        return sum(numbers)
    if op == "mul":
        product = 1.0
        for number in numbers:
            product *= number
        return product
    if op == "sub":
        result = numbers[0]
        for number in numbers[1:]:
            result -= number
        return result
    if op == "div":
        result = numbers[0]
        for number in numbers[1:]:
            if number == 0:
                return None
            result /= number
        return result
    return None


class ActionCalculateMath(Action):
    def name(self) -> Text:
        return "action_calculate_math"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, object]]:
        text = (tracker.latest_message.get("text") or "").strip()
        if not text:
            dispatcher.utter_message(response="utter_ask_math")
            return []

        expression = parse_expression(text)
        if expression:
            result = calculate(expression["op"], [expression["left"], expression["right"]])
            if result is None:
                dispatcher.utter_message(text="0-д хуваах боломжгүй.")
                return []
            left_text = format_number(expression["left"])
            right_text = format_number(expression["right"])
            result_text = format_number(result)
            dispatcher.utter_message(
                text=f"Тооцоолол: {left_text} {expression['symbol']} {right_text} = {result_text}"
            )
            dispatcher.utter_message(text=f"Хариу: {result_text}")
            return []

        numbers = extract_numbers(text)
        if len(numbers) < 2:
            dispatcher.utter_message(response="utter_ask_math")
            return []

        operation = detect_operation(text)
        if not operation:
            dispatcher.utter_message(response="utter_invalid_math")
            return []

        result = calculate(operation, numbers)
        if result is None:
            dispatcher.utter_message(text="0-д хуваах боломжгүй.")
            return []

        symbol_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        symbol = symbol_map.get(operation, "?")
        expression_text = f" {symbol} ".join(format_number(number) for number in numbers)
        result_text = format_number(result)
        dispatcher.utter_message(text=f"Тооцоолол: {expression_text} = {result_text}")
        dispatcher.utter_message(text=f"Хариу: {result_text}")
        return []
