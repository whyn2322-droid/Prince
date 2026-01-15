from typing import Any, Dict, List, Text, Optional
import math
import re

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    SYMPY_AVAILABLE = True
    SYMPY_TRANSFORMS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    SYMPY_X = sp.symbols("x")
    SYMPY_LOCALS = {
        "x": SYMPY_X,
        "e": sp.E,
        "E": sp.E,
        "pi": sp.pi,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "sec": sp.sec,
        "csc": sp.csc,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
    }
except Exception:
    sp = None
    parse_expr = None
    SYMPY_AVAILABLE = False
    SYMPY_TRANSFORMS = None
    SYMPY_X = None
    SYMPY_LOCALS = {}

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import AllSlotsReset, FollowupAction, SlotSet

NUMBER_MESSAGE = "Зөв тоо оруулна уу. (ж: 2, -3, 4.5)"
POSITIVE_MESSAGE = "Эерэг тоо оруулна уу. (ж: 2, 4.5)"
INTEGER_MESSAGE = "Бүхэл тоо оруулна уу. (ж: 3, 12)"
NONNEG_INTEGER_MESSAGE = "0 эсвэл эерэг бүхэл тоо оруулна уу. (ж: 0, 5)"
PROBABILITY_MESSAGE = "0–1 эсвэл 0–100 хоорондын утга оруулна уу. (ж: 0.3 эсвэл 30)"


def _parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _is_int(value: float) -> bool:
    return abs(value - round(value)) < 1e-9


def _to_int(value: float) -> int:
    return int(round(value))


def _fmt_num(value: float) -> str:
    return f"{value:g}"


def _strip_outer_parens(text: str) -> str:
    if not text.startswith("(") or not text.endswith(")"):
        return text
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return text
    if depth == 0:
        return text[1:-1]
    return text


def _normalize_expr(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = t.replace(" ", "")
    t = t.replace("\u00d7", "*").replace("\u00f7", "/")
    t = t.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\u222b", "")
    t = t.replace("dx", "")
    t = re.sub(r"^int", "", t)
    t = re.sub(r"^integral", "", t)
    t = re.sub(r"\bsqrt(\d+(?:\.\d+)?)\b", r"sqrt(\1)", t)
    t = re.sub(
        r"\b(sin|cos|tan|cot|sec|csc|asin|acos|atan|sinh|cosh|tanh|exp|ln|log)(?!\s*\()([a-zA-Z]\w*|\d+(?:\.\d+)?)",
        r"\1(\2)",
        t,
    )
    t = re.sub(r"sqrtx", "x^0.5", t)
    t = t.replace("sqrt(x)", "x^0.5")
    t = re.sub(r"x(\d+)", r"x^\1", t)
    t = _strip_outer_parens(t)
    return t

def _extract_integrand(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^(find|compute|evaluate|solve|integrate)\s*:?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^i\s*=\s*", "", t, flags=re.IGNORECASE)
    t = t.replace("\u222b", "")
    t = re.sub(r"^int\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^integral\s*", "", t, flags=re.IGNORECASE)
    t = re.split(r"\bdx\b", t, maxsplit=1, flags=re.IGNORECASE)[0]
    if "=" in t:
        t = t.split("=", 1)[0]
    t = t.strip().strip(" .;")
    return t


def _sympy_parse(text: str):
    if not SYMPY_AVAILABLE:
        return None
    raw = _extract_integrand(text)
    if not raw:
        return None
    norm = _normalize_expr(raw)
    try:
        expr = parse_expr(
            norm,
            local_dict=SYMPY_LOCALS,
            transformations=SYMPY_TRANSFORMS,
            evaluate=True,
        )
    except Exception:
        return None
    return expr, norm


def _sympy_integrate(text: str):
    if not SYMPY_AVAILABLE:
        return None
    parsed = _sympy_parse(text)
    if not parsed:
        return None
    expr, norm = parsed
    try:
        result = sp.integrate(expr, SYMPY_X)
    except Exception:
        return None
    if isinstance(result, sp.Integral) or result.has(sp.Integral):
        return None
    result_str = str(result)
    result_str = result_str.replace("**", "^")
    result_str = re.sub(r"\bE\b", "e", result_str)
    result_str = re.sub(r"\blog\b", "ln", result_str)
    return result_str, norm



def _parse_exponent(exp_str: str) -> Optional[float]:
    exp = exp_str.strip()
    if exp.startswith("(") and exp.endswith(")"):
        exp = exp[1:-1]
    if "/" in exp:
        num_str, den_str = exp.split("/", 1)
        den = float(den_str)
        if den == 0:
            return None
        return float(num_str) / den
    return float(exp)


def _split_terms(expr: str) -> List[str]:
    terms: List[str] = []
    buf = ""
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if depth == 0 and ch in "+-" and buf:
            terms.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        terms.append(buf)
    return terms


def _parse_simple_x_term(term: str) -> Optional[tuple[float, float]]:
    if term.startswith("x"):
        coeff = 1.0
        rest = term[1:]
    else:
        m = re.match(r"([0-9]+(?:\.[0-9]+)?)(?:\*?)x(.*)$", term)
        if not m:
            return None
        coeff = float(m.group(1))
        rest = m.group(2)
    exp = 1.0
    if rest.startswith("^"):
        exp = _parse_exponent(rest[1:])
        if exp is None:
            return None
    elif rest:
        return None
    return coeff, exp


def _parse_term(term: str) -> Optional[tuple[float, float]]:
    if not term:
        return None
    sign = 1.0
    if term[0] == "+":
        term = term[1:]
    elif term[0] == "-":
        sign = -1.0
        term = term[1:]
    if not term:
        return None

    if term.startswith("1/(") and term.endswith(")"):
        inner = term[3:-1]
        inner_parsed = _parse_simple_x_term(inner)
        if inner_parsed is None:
            return None
        coeff_inner, exp_inner = inner_parsed
        if coeff_inner == 0:
            return None
        return sign * (1.0 / coeff_inner), -exp_inner

    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\/x(?:\^(.+))?", term)
    if m:
        coeff = float(m.group(1))
        exp = 1.0
        if m.group(2):
            exp = _parse_exponent(m.group(2))
            if exp is None:
                return None
        return sign * coeff, -exp

    if "x" not in term:
        try:
            return sign * float(term), 0.0
        except Exception:
            return None

    simple = _parse_simple_x_term(term)
    if simple is None:
        return None
    coeff, exp = simple
    return sign * coeff, exp


def _combine_terms(terms: List[tuple[float, float]]) -> List[tuple[float, float]]:
    combined: Dict[float, float] = {}
    for coeff, exp in terms:
        key = round(exp, 10)
        combined[key] = combined.get(key, 0.0) + coeff
    result = [(coeff, exp) for exp, coeff in combined.items() if abs(coeff) > 1e-12]
    result.sort(key=lambda t: t[1], reverse=True)
    return result


def _parse_expression(text: str) -> Optional[List[tuple[float, float]]]:
    expr = _normalize_expr(text)
    if not expr:
        return None
    raw_terms = _split_terms(expr)
    terms: List[tuple[float, float]] = []
    for raw in raw_terms:
        parsed = _parse_term(raw)
        if parsed is None:
            return None
        terms.append(parsed)
    return _combine_terms(terms)


def _format_power_term(coeff: float, exp: float) -> str:
    coeff_abs = abs(coeff)
    if abs(exp) < 1e-9:
        return _fmt_num(coeff_abs)
    if abs(exp - 1.0) < 1e-9:
        base = "x"
    else:
        base = f"x^{_fmt_num(exp)}"
    if abs(coeff_abs - 1.0) < 1e-9:
        return base
    return f"{_fmt_num(coeff_abs)}*{base}"


def _format_integral(terms: List[tuple[float, float]]) -> str:
    parts: List[tuple[int, str]] = []
    for coeff, exp in terms:
        if abs(coeff) < 1e-12:
            continue
        if abs(exp + 1.0) < 1e-9:
            coeff_abs = abs(coeff)
            body = "ln|x|" if abs(coeff_abs - 1.0) < 1e-9 else f"{_fmt_num(coeff_abs)}*ln|x|"
            parts.append((1 if coeff >= 0 else -1, body))
            continue
        new_exp = exp + 1.0
        new_coeff = coeff / (exp + 1.0)
        body = _format_power_term(new_coeff, new_exp)
        parts.append((1 if new_coeff >= 0 else -1, body))

    if not parts:
        return "0"

    result = ""
    for sign, body in parts:
        if not result:
            result = f"-{body}" if sign < 0 else body
        else:
            op = "-" if sign < 0 else "+"
            result += f" {op} {body}"
    return result


def _get_slot_float(tracker: Tracker, slot_name: Text) -> Optional[float]:
    value = tracker.get_slot(slot_name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_numbers(
    dispatcher: CollectingDispatcher, tracker: Tracker, slot_names: List[Text]
) -> Optional[List[float]]:
    values: List[float] = []
    for slot_name in slot_names:
        num = _get_slot_float(tracker, slot_name)
        if num is None:
            dispatcher.utter_message(text=NUMBER_MESSAGE)
            return None
        values.append(num)
    return values


def _require_positive_numbers(
    dispatcher: CollectingDispatcher, tracker: Tracker, slot_names: List[Text]
) -> Optional[List[float]]:
    values = _require_numbers(dispatcher, tracker, slot_names)
    if values is None:
        return None
    for num in values:
        if num <= 0:
            dispatcher.utter_message(text=POSITIVE_MESSAGE)
            return None
    return values


def _require_nonneg_ints(
    dispatcher: CollectingDispatcher, tracker: Tracker, slot_names: List[Text]
) -> Optional[List[int]]:
    values: List[int] = []
    for slot_name in slot_names:
        num = _get_slot_float(tracker, slot_name)
        if num is None or not _is_int(num):
            dispatcher.utter_message(text=INTEGER_MESSAGE)
            return None
        val = _to_int(num)
        if val < 0:
            dispatcher.utter_message(text=NONNEG_INTEGER_MESSAGE)
            return None
        values.append(val)
    return values


def _normalize_prob(value: float) -> Optional[float]:
    if value < 0:
        return None
    if value <= 1:
        return value
    if value <= 100:
        return value / 100.0
    return None


def _factorial(n: int) -> int:
    return math.factorial(n)


def _perm(n: int, r: int) -> int:
    if hasattr(math, "perm"):
        return math.perm(n, r)
    return math.factorial(n) // math.factorial(n - r)


def _comb(n: int, r: int) -> int:
    if hasattr(math, "comb"):
        return math.comb(n, r)
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))


class ValidateMath12Form(FormValidationAction):
    def name(self) -> Text:
        return "validate_math12_form"

    async def required_slots(
        self,
        slots_mapped_in_domain: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Text]:
        topic = tracker.get_slot("topic")

        if not topic:
            return ["topic"]

        if topic == "integral":
            func = tracker.get_slot("integral_func")
            kind = tracker.get_slot("integral_kind")

            if func == "expression":
                return ["topic", "integral_func", "int_expr"]

            slots = ["topic", "integral_kind", "integral_func", "int_coeff"]
            if func == "power":
                slots.append("int_power")
            if kind == "definite":
                slots.extend(["int_a", "int_b"])
            return slots

        if topic == "analysis":
            return ["topic", "quad_a", "quad_b", "quad_c"]

        if topic == "parameter":
            return ["topic", "param_a", "param_b"]

        if topic == "combinatorics":
            slots = ["topic", "comb_type", "comb_n"]
            if tracker.get_slot("comb_type") in ["permutation", "combination"]:
                slots.append("comb_r")
            return slots

        if topic == "probability":
            return ["topic", "prob_a", "prob_b", "prob_intersection"]

        return ["topic"]

    def validate_topic(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Please choose a topic.")
            return {"topic": None}

        v = value.strip().lower()
        mapping = {
            "integral": "integral",
            "analysis": "analysis",
            "parameter": "parameter",
            "combinatorics": "combinatorics",
            "probability": "probability",
        }

        topic = mapping.get(v, v)
        allowed = {"integral", "analysis", "parameter", "combinatorics", "probability"}
        if topic not in allowed:
            dispatcher.utter_message(text="Unknown topic. Choose again.")
            return {"topic": None}

        return {
            "topic": topic,
            "integral_kind": None,
            "integral_func": None,
            "int_coeff": None,
            "int_power": None,
            "int_a": None,
            "int_b": None,
            "int_expr": None,
            "quad_a": None,
            "quad_b": None,
            "quad_c": None,
            "param_a": None,
            "param_b": None,
            "comb_type": None,
            "comb_n": None,
            "comb_r": None,
            "prob_a": None,
            "prob_b": None,
            "prob_intersection": None,
        }

    def validate_integral_kind(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Choose integral kind.")
            return {"integral_kind": None}

        v = value.strip().lower()
        mapping = {
            "indefinite": "indefinite",
            "definite": "definite",
        }

        kind = mapping.get(v, v)
        if kind not in {"indefinite", "definite"}:
            dispatcher.utter_message(text="Unknown integral kind.")
            return {"integral_kind": None}

        return {"integral_kind": kind, "int_a": None, "int_b": None}

    def validate_integral_func(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Choose integral function type.")
            return {"integral_func": None}

        v = value.strip().lower()
        mapping = {
            "x^n": "power",
            "a*x^n": "power",
            "power": "power",
            "1/x": "reciprocal",
            "a/x": "reciprocal",
            "reciprocal": "reciprocal",
            "expression": "expression",
        }

        func = mapping.get(v, v)
        if func not in {"power", "reciprocal", "expression"}:
            dispatcher.utter_message(text="Unknown function type.")
            return {"integral_func": None}

        if func == "expression":
            return {
                "integral_func": func,
                "integral_kind": "indefinite",
                "int_expr": None,
                "int_coeff": None,
                "int_power": None,
                "int_a": None,
                "int_b": None,
            }

        return {"integral_func": func, "int_power": None}

    def validate_comb_type(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Choose combinatorics type.")
            return {"comb_type": None}

        v = value.strip().lower()
        mapping = {
            "factorial": "factorial",
            "permutation": "permutation",
            "p(n,r)": "permutation",
            "combination": "combination",
            "c(n,r)": "combination",
        }

        comb_type = mapping.get(v, v)
        if comb_type not in {"factorial", "permutation", "combination"}:
            dispatcher.utter_message(text="Unknown combinatorics type.")
            return {"comb_type": None}

        return {"comb_type": comb_type, "comb_n": None, "comb_r": None}

    def _validate_number(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if num is None:
            dispatcher.utter_message(text=NUMBER_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(num)}

    def _validate_positive(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if num is None or num <= 0:
            dispatcher.utter_message(text=POSITIVE_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(num)}

    def _validate_nonzero(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if num is None or num == 0:
            dispatcher.utter_message(text="Value must be non-zero.")
            return {slot_name: None}
        return {slot_name: float(num)}

    def _validate_int(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if num is None or not _is_int(num):
            dispatcher.utter_message(text=INTEGER_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(_to_int(num))}

    def _validate_nonneg_int(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if num is None or not _is_int(num):
            dispatcher.utter_message(text=INTEGER_MESSAGE)
            return {slot_name: None}
        val = _to_int(num)
        if val < 0:
            dispatcher.utter_message(text=NONNEG_INTEGER_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(val)}

    def _validate_prob(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if num is None:
            dispatcher.utter_message(text=NUMBER_MESSAGE)
            return {slot_name: None}
        if num < 0 or num > 100:
            dispatcher.utter_message(text=PROBABILITY_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(num)}

    def validate_int_coeff(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("int_coeff", dispatcher, value)

    def validate_int_power(self, value: Any, dispatcher, tracker, domain):
        num = _parse_number(str(value))
        if num is None or not _is_int(num):
            dispatcher.utter_message(text=INTEGER_MESSAGE)
            return {"int_power": None}
        if _to_int(num) == -1:
            dispatcher.utter_message(text="n cannot be -1. Use 1/x form instead.")
            return {"int_power": None}
        return {"int_power": float(_to_int(num))}

    def validate_int_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("int_a", dispatcher, value)

    def validate_int_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("int_b", dispatcher, value)

    def validate_int_expr(self, value: Any, dispatcher, tracker, domain):
        expr = "" if value is None else str(value)
        if _sympy_parse(expr) or _parse_expression(expr):
            return {"int_expr": expr}
        if SYMPY_AVAILABLE:
            dispatcher.utter_message(
                text="Expression not recognized. Examples: x*e^x, x*sin(x), x^2*ln(x), 1/(x^2-1)"
            )
        else:
            dispatcher.utter_message(text="Sympy not installed. Run: pip install sympy")
        return {"int_expr": None}

    def validate_quad_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_nonzero("quad_a", dispatcher, value)

    def validate_quad_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("quad_b", dispatcher, value)

    def validate_quad_c(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("quad_c", dispatcher, value)

    def validate_param_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("param_a", dispatcher, value)

    def validate_param_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("param_b", dispatcher, value)

    def validate_comb_n(self, value: Any, dispatcher, tracker, domain):
        return self._validate_nonneg_int("comb_n", dispatcher, value)

    def validate_comb_r(self, value: Any, dispatcher, tracker, domain):
        return self._validate_nonneg_int("comb_r", dispatcher, value)

    def validate_prob_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_prob("prob_a", dispatcher, value)

    def validate_prob_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_prob("prob_b", dispatcher, value)

    def validate_prob_intersection(self, value: Any, dispatcher, tracker, domain):
        return self._validate_prob("prob_intersection", dispatcher, value)


class ActionCalculateMath12(Action):
    def name(self) -> Text:
        return "action_calculate_math12"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        topic = tracker.get_slot("topic")

        if topic == "integral":
            kind = tracker.get_slot("integral_kind")
            func = tracker.get_slot("integral_func")

            if func == "expression":
                expr = tracker.get_slot("int_expr")
                expr_str = "" if expr is None else str(expr)
                sympy_result = _sympy_integrate(expr_str)
                if sympy_result:
                    result_str, norm = sympy_result
                    dispatcher.utter_message(text=f"Integral: int ({norm}) dx = {result_str} + C")
                    return []
                parsed = _parse_expression(expr_str)
                if parsed:
                    result = _format_integral(parsed)
                    norm = _normalize_expr(expr_str)
                    dispatcher.utter_message(text=f"Integral: int ({norm}) dx = {result} + C")
                    return []
                if SYMPY_AVAILABLE:
                    dispatcher.utter_message(
                        text="Expression not recognized. Examples: x*e^x, x*sin(x), x^2*ln(x), 1/(x^2-1)"
                    )
                else:
                    dispatcher.utter_message(text="Sympy not installed. Run: pip install sympy")
                return []

            values = _require_numbers(dispatcher, tracker, ["int_coeff"])
            if not values:
                return []
            a = values[0]

            if func == "power":
                n_val = _get_slot_float(tracker, "int_power")
                if n_val is None or not _is_int(n_val) or _to_int(n_val) == -1:
                    dispatcher.utter_message(text="Invalid n. Use integer n != -1.")
                    return []
                n = _to_int(n_val)
                coef = a / (n + 1)

                if kind == "indefinite":
                    dispatcher.utter_message(
                        text=(
                            "Integral: int a*x^n dx = (a/(n+1)) * x^(n+1) + C. "
                            f"int {_fmt_num(a)}*x^{n} dx = {_fmt_num(coef)}*x^{n + 1} + C"
                        )
                    )
                    return []

                if kind == "definite":
                    bounds = _require_numbers(dispatcher, tracker, ["int_a", "int_b"])
                    if not bounds:
                        return []
                    x1, x2 = bounds
                    try:
                        f_x2 = coef * (x2 ** (n + 1))
                        f_x1 = coef * (x1 ** (n + 1))
                    except Exception:
                        dispatcher.utter_message(text="Invalid bounds.")
                        return []
                    area = f_x2 - f_x1
                    dispatcher.utter_message(
                        text=(
                            "Definite integral: F(x) = (a/(n+1))*x^(n+1). "
                            f"S = F(b) - F(a) = {_fmt_num(f_x2)} - {_fmt_num(f_x1)} = {_fmt_num(area)}"
                        )
                    )
                    return []

                dispatcher.utter_message(text="Unknown integral kind.")
                return []

            if func == "reciprocal":
                if kind == "indefinite":
                    dispatcher.utter_message(
                        text=(
                            "Integral: int a/x dx = a*ln|x| + C. "
                            f"int {_fmt_num(a)}/x dx = {_fmt_num(a)}*ln|x| + C"
                        )
                    )
                    return []

                if kind == "definite":
                    bounds = _require_numbers(dispatcher, tracker, ["int_a", "int_b"])
                    if not bounds:
                        return []
                    x1, x2 = bounds
                    if x1 == 0 or x2 == 0:
                        dispatcher.utter_message(text="1/x is undefined at 0.")
                        return []
                    area = a * (math.log(abs(x2)) - math.log(abs(x1)))
                    dispatcher.utter_message(
                        text=(
                            "Definite integral: S = a*(ln|b| - ln|a|). "
                            f"S = {_fmt_num(a)}*(ln|{_fmt_num(x2)}| - ln|{_fmt_num(x1)}|) = {_fmt_num(area)}"
                        )
                    )
                    return []

                dispatcher.utter_message(text="Unknown integral kind.")
                return []

            dispatcher.utter_message(text="Unknown integral function.")
            return []

        if topic == "analysis":
            values = _require_numbers(dispatcher, tracker, ["quad_a", "quad_b", "quad_c"])
            if not values:
                return []
            a, b, c = values
            if a == 0:
                dispatcher.utter_message(text="a cannot be 0.")
                return []
            x0 = -b / (2 * a)
            y0 = a * x0 * x0 + b * x0 + c
            d = b * b - 4 * a * c
            direction = "up" if a > 0 else "down"
            extremum = "min" if a > 0 else "max"
            if d >= 0:
                sqrt_d = math.sqrt(d)
                x1 = (-b - sqrt_d) / (2 * a)
                x2 = (-b + sqrt_d) / (2 * a)
                roots = f"x1 = {_fmt_num(x1)}, x2 = {_fmt_num(x2)}"
            else:
                roots = "no real roots"
            if a > 0:
                dec_int = f"(-inf, {_fmt_num(x0)})"
                inc_int = f"({_fmt_num(x0)}, +inf)"
            else:
                inc_int = f"(-inf, {_fmt_num(x0)})"
                dec_int = f"({_fmt_num(x0)}, +inf)"
            dispatcher.utter_message(
                text=(
                    f"f(x) = {_fmt_num(a)}x^2 + {_fmt_num(b)}x + {_fmt_num(c)}\n"
                    f"vertex: ({_fmt_num(x0)}, {_fmt_num(y0)})\n"
                    f"opens {direction}, {extremum} = {_fmt_num(y0)}\n"
                    f"roots: {roots}\n"
                    f"decreasing: {dec_int}, increasing: {inc_int}"
                )
            )
            return []

        if topic == "parameter":
            values = _require_numbers(dispatcher, tracker, ["param_a", "param_b"])
            if not values:
                return []
            a, b = values
            if a == 0 and b == 0:
                dispatcher.utter_message(text="ax + b = 0 has infinite solutions.")
                return []
            if a == 0 and b != 0:
                dispatcher.utter_message(text="ax + b = 0 has no solution.")
                return []
            x = -b / a
            dispatcher.utter_message(text=f"ax + b = 0 => x = -b/a = {_fmt_num(x)}")
            return []

        if topic == "combinatorics":
            comb_type = tracker.get_slot("comb_type")
            if comb_type == "factorial":
                values = _require_nonneg_ints(dispatcher, tracker, ["comb_n"])
                if not values:
                    return []
                n = values[0]
                result = _factorial(n)
                dispatcher.utter_message(text=f"{n}! = {result}")
                return []

            if comb_type in ["permutation", "combination"]:
                values = _require_nonneg_ints(dispatcher, tracker, ["comb_n", "comb_r"])
                if not values:
                    return []
                n, r = values
                if r > n:
                    dispatcher.utter_message(text="r must be <= n.")
                    return []
                if comb_type == "permutation":
                    result = _perm(n, r)
                    dispatcher.utter_message(text=f"P(n,r) = n!/(n-r)! = {result}")
                    return []
                result = _comb(n, r)
                dispatcher.utter_message(text=f"C(n,r) = n!/(r!(n-r)!) = {result}")
                return []

            dispatcher.utter_message(text="Unknown combinatorics type.")
            return []

        if topic == "probability":
            values = _require_numbers(dispatcher, tracker, ["prob_a", "prob_b", "prob_intersection"])
            if not values:
                return []
            pa_raw, pb_raw, pab_raw = values
            pa = _normalize_prob(pa_raw)
            pb = _normalize_prob(pb_raw)
            pab = _normalize_prob(pab_raw)
            if pa is None or pb is None or pab is None:
                dispatcher.utter_message(text=PROBABILITY_MESSAGE)
                return []
            p_union = pa + pb - pab
            percent = p_union * 100
            if p_union < 0 or p_union > 1:
                dispatcher.utter_message(text="Invalid probability values.")
            dispatcher.utter_message(
                text=(
                    "P(A U B) = P(A) + P(B) - P(A U B) = "
                    f"{_fmt_num(pa)} + {_fmt_num(pb)} - {_fmt_num(pab)} = {_fmt_num(p_union)}"
                    f" ({_fmt_num(percent)}%)"
                )
            )
            return []

        dispatcher.utter_message(text="Unknown topic.")
        return []


class ActionResetMath12(Action):
    def name(self) -> Text:
        return "action_reset_math12"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        last_intent = (tracker.latest_message or {}).get("intent", {}).get("name")
        if last_intent == "restart_math":
            return [AllSlotsReset()]

        if tracker.get_slot("topic") == "integral" and tracker.get_slot("integral_func") == "expression":
            return [SlotSet("int_expr", None), FollowupAction("math12_form")]

        return [AllSlotsReset()]
