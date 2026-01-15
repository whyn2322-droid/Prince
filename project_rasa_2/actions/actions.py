from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_sdk import Action, FormValidationAction, Tracker
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from sympy import Abs, Eq, cancel, expand, simplify, solve
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.solvers.inequalities import solve_univariate_inequality

EPSILON = 1e-9

NUMBER_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)?")
SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
}
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def normalize_input(text: Text) -> Text:
    replacements = {
        "×": "*",
        "∙": "*",
        "·": "*",
        "−": "-",
        "–": "-",
        "—": "-",
        "≤": "<=",
        "≥": ">=",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def replace_superscripts(text: Text) -> Text:
    result: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in SUPERSCRIPT_MAP:
            digits: List[str] = []
            while i < len(text) and text[i] in SUPERSCRIPT_MAP:
                digits.append(SUPERSCRIPT_MAP[text[i]])
                i += 1
            result.append("**" + "".join(digits))
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def convert_abs_bars(text: Text) -> Text:
    result: List[str] = []
    open_abs = False
    for ch in text:
        if ch == "|":
            if open_abs:
                result.append(")")
            else:
                result.append("Abs(")
            open_abs = not open_abs
        else:
            result.append(ch)
    if open_abs:
        raise ValueError("Unclosed absolute value")
    return "".join(result)


def find_matching_paren(text: Text, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def convert_sqrt(text: Text) -> Text:
    result: List[str] = []
    i = 0
    while i < len(text):
        if text[i] != "√":
            result.append(text[i])
            i += 1
            continue

        i += 1
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            raise ValueError("Missing sqrt argument")

        if text[i] == "(":
            end = find_matching_paren(text, i)
            if end == -1:
                raise ValueError("Missing closing parenthesis")
            inner = text[i + 1 : end]
            result.append(f"sqrt({inner})")
            i = end + 1
            continue

        start = i
        depth = 0
        while i < len(text):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            if depth == 0 and ch in "+-*/=<>|,":
                break
            i += 1
        inner = text[start:i].strip()
        if not inner:
            raise ValueError("Missing sqrt argument")
        result.append(f"sqrt({inner})")
    return "".join(result)


def normalize_math_input(text: Text) -> Text:
    text = normalize_input(text)
    text = replace_superscripts(text)
    text = convert_abs_bars(text)
    text = convert_sqrt(text)
    return text


def parse_expression(text: Text):
    normalized = normalize_math_input(text)
    local_dict = {"sqrt": __import__("sympy").sqrt, "Abs": Abs}
    return parse_expr(normalized, local_dict=local_dict, transformations=TRANSFORMATIONS)


def format_expr(expr) -> str:
    text = str(expr)
    text = text.replace("**", "^")
    text = text.replace("sqrt(", "√(")
    return text


def format_number(value: float) -> str:
    if abs(value - round(value)) < EPSILON:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def parse_number(text: Text) -> Optional[float]:
    match = NUMBER_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def split_equation(text: Text) -> Optional[Tuple[Text, Text]]:
    if "=" not in text:
        return None
    parts = text.split("=")
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def split_inequality(text: Text) -> Optional[Tuple[Text, Text, Text]]:
    text = normalize_input(text)
    for op in [">=", "<=", ">", "<"]:
        if op in text:
            parts = text.split(op)
            if len(parts) != 2:
                return None
            return parts[0].strip(), op, parts[1].strip()
    return None


def pick_symbol(symbols: List) -> Optional:
    if not symbols:
        return None
    return sorted(symbols, key=lambda s: s.name)[0]


def denominator_values(expr) -> List[str]:
    denom = __import__("sympy").denom(expr)
    if denom == 1:
        return []
    symbols = list(denom.free_symbols)
    if len(symbols) != 1:
        return []
    var = symbols[0]
    roots = solve(Eq(denom, 0), var)
    return [f"{var} ≠ {format_expr(root)}" for root in roots]


class ValidatePolygonForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_polygon_form"

    def validate_polygon_sides(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_number(value)
        if number is None:
            dispatcher.utter_message(text="3-аас их бүхэл тоо оруулна уу.")
            return {"polygon_sides": None}
        sides = int(number)
        if abs(number - sides) > EPSILON or sides < 3:
            dispatcher.utter_message(text="3-аас их бүхэл тоо оруулна уу.")
            return {"polygon_sides": None}
        return {"polygon_sides": str(sides)}


class ActionSimplifyExpression(Action):
    def name(self) -> Text:
        return "action_simplify_expression"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        expression = tracker.get_slot("simplify_expression") or ""
        try:
            expr = parse_expression(expression)
        except Exception:
            dispatcher.utter_message(text="Илэрхийлэлээ зөв бичнэ үү. Ж: (x+4)-(2x-1)")
            return [AllSlotsReset()]

        expanded = expand(expr)
        simplified = simplify(expanded)
        reduced = cancel(simplified)
        final_expr = reduced if len(str(reduced)) <= len(str(simplified)) else simplified

        if expanded != expr:
            dispatcher.utter_message(text=f"Хаалт задлав: {format_expr(expanded)}")
        if final_expr != expanded:
            dispatcher.utter_message(text=f"Нэгтгэсэн: {format_expr(final_expr)}")

        dispatcher.utter_message(text=f"Хялбаршуулсан хэлбэр: {format_expr(final_expr)}")
        restrictions = denominator_values(expr)
        if restrictions:
            dispatcher.utter_message(text=f"Хязгаарлалт: {', '.join(restrictions)}")
        return [AllSlotsReset()]


class ActionExpandBrackets(Action):
    def name(self) -> Text:
        return "action_expand_brackets"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        expression = tracker.get_slot("expand_expression") or ""
        try:
            expr = parse_expression(expression)
        except Exception:
            dispatcher.utter_message(text="Илэрхийлэлээ зөв бичнэ үү. Ж: 3(x-2)")
            return [AllSlotsReset()]

        expanded = expand(expr)
        dispatcher.utter_message(text="Томьёо: a(b+c)=ab+ac, a(b-c)=ab-ac.")
        dispatcher.utter_message(text=f"Өргөтгөсөн хэлбэр: {format_expr(expanded)}")
        return [AllSlotsReset()]


class ActionCombineLikeTerms(Action):
    def name(self) -> Text:
        return "action_combine_like_terms"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        expression = tracker.get_slot("combine_expression") or ""
        try:
            expr = parse_expression(expression)
        except Exception:
            dispatcher.utter_message(text="Илэрхийлэлээ зөв бичнэ үү. Ж: 2x+3x-x+5")
            return [AllSlotsReset()]

        combined = simplify(expand(expr))
        dispatcher.utter_message(text="Тайлбар: адил гишүүдийг нийлүүлнэ.")
        dispatcher.utter_message(text=f"Нэгтгэсэн хэлбэр: {format_expr(combined)}")
        return [AllSlotsReset()]


class ActionSolveLinearEquation(Action):
    def name(self) -> Text:
        return "action_solve_linear_equation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        equation = tracker.get_slot("equation") or ""
        parts = split_equation(equation)
        if not parts:
            dispatcher.utter_message(text="Жишээ: 2x+3=7")
            return [AllSlotsReset()]

        left_text, right_text = parts
        try:
            left = parse_expression(left_text)
            right = parse_expression(right_text)
        except Exception:
            dispatcher.utter_message(text="Тэгшитгэлээ зөв бичнэ үү. Ж: 2x+3=7")
            return [AllSlotsReset()]

        symbols = list((left - right).free_symbols)
        var = pick_symbol(symbols)
        if var is None:
            if simplify(left - right) == 0:
                dispatcher.utter_message(text="Бүх бодит тоо шийд болно.")
            else:
                dispatcher.utter_message(text="Шийд байхгүй.")
            return [AllSlotsReset()]
        if len(symbols) > 1:
            dispatcher.utter_message(
                text="Нэг л хувьсагчтай тэгшитгэл оруулна уу. Ж: 2x+3=7"
            )
            return [AllSlotsReset()]

        dispatcher.utter_message(
            text=f"Нэг талд шилжүүлбэл: {format_expr(simplify(left - right))} = 0"
        )
        solutions = solve(Eq(left, right), var)
        if not solutions:
            dispatcher.utter_message(text="Шийд байхгүй.")
            return [AllSlotsReset()]

        if len(solutions) == 1:
            dispatcher.utter_message(
                text=f"Шийд: {var} = {format_expr(solutions[0])}"
            )
        else:
            solution_text = ", ".join(format_expr(sol) for sol in solutions)
            dispatcher.utter_message(text=f"Шийдүүд: {var} = {solution_text}")
        return [AllSlotsReset()]


class ActionSolveInequality(Action):
    def name(self) -> Text:
        return "action_solve_inequality"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        inequality = tracker.get_slot("inequality") or ""
        parts = split_inequality(inequality)
        if not parts:
            dispatcher.utter_message(text="Жишээ: -2x+1>5")
            return [AllSlotsReset()]

        left_text, op, right_text = parts
        try:
            left = parse_expression(left_text)
            right = parse_expression(right_text)
        except Exception:
            dispatcher.utter_message(text="Тэнцэтгэл бишээ зөв бичнэ үү. Ж: -2x+1>5")
            return [AllSlotsReset()]

        symbols = list((left - right).free_symbols)
        var = pick_symbol(symbols)
        if var is None:
            truth = simplify(left - right)
            satisfied = {
                ">": truth > 0,
                "<": truth < 0,
                ">=": truth >= 0,
                "<=": truth <= 0,
            }[op]
            dispatcher.utter_message(
                text="Бүх бодит тоо шийд болно." if satisfied else "Шийд байхгүй."
            )
            return [AllSlotsReset()]
        if len(symbols) > 1:
            dispatcher.utter_message(
                text="Нэг л хувьсагчтай тэнцэтгэл биш оруулна уу. Ж: -2x+1>5"
            )
            return [AllSlotsReset()]

        relation = {
            ">": left > right,
            "<": left < right,
            ">=": left >= right,
            "<=": left <= right,
        }[op]
        solution = solve_univariate_inequality(relation, var, relational=True)
        dispatcher.utter_message(
            text=f"Шийд: {format_expr(solution)}"
        )
        return [AllSlotsReset()]


class ActionPolygonAngles(Action):
    def name(self) -> Text:
        return "action_polygon_angles"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        sides_text = tracker.get_slot("polygon_sides") or ""
        number = parse_number(sides_text)
        if number is None:
            dispatcher.utter_message(text="3-аас их бүхэл тоо оруулна уу.")
            return [AllSlotsReset()]

        sides = int(number)
        if abs(number - sides) > EPSILON or sides < 3:
            dispatcher.utter_message(text="3-аас их бүхэл тоо оруулна уу.")
            return [AllSlotsReset()]

        sum_angles = (sides - 2) * 180
        regular_angle = sum_angles / sides

        dispatcher.utter_message(text="Томьёо: дотоод өнцгүүдийн нийлбэр = (n-2) * 180.")
        dispatcher.utter_message(
            text=f"Нийлбэр: ({sides}-2) * 180 = {format_number(sum_angles)} градус"
        )
        dispatcher.utter_message(
            text=(
                "Тэгш олон өнцөгт бол нэг дотоод өнцөг: "
                f"{format_number(regular_angle)} градус"
            )
        )
        dispatcher.utter_message(text="Гадаад өнцгүүдийн нийлбэр = 360 градус.")
        return [AllSlotsReset()]
