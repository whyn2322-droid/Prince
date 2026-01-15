from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Text, Tuple

import sympy as sp
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

EPSILON = 1e-9
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
TUPLE_RE = re.compile(r"\(([^)]+)\)")
PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
MATH_RE = re.compile(r"[A-Za-z0-9^*+\-()/\.]+")
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
MATH_FUNCS = ("sin", "cos", "tan", "sqrt", "ln", "log")
SUGGESTION_TEXT = (
    "Санал болгох сэдвүүд: шулуун, зай/дунд цэг, тойрог, вектор, "
    "уламжлал/предел, өсөх-буурах/экстремум, магадлал."
)

LOOKALIKE_TRANSLATION = str.maketrans({
    "a": "?",
    "b": "?",
    "c": "?",
    "e": "?",
    "h": "?",
    "i": "?",
    "k": "?",
    "l": "?",
    "m": "?",
    "n": "?",
    "o": "?",
    "p": "?",
    "r": "?",
    "t": "?",
    "u": "?",
    "x": "?",
    "y": "?",
})


def parse_numbers(text: Text) -> List[float]:
    return [float(n.replace(",", ".")) for n in NUMBER_RE.findall(text)]


def format_number(value: float) -> str:
    if abs(value - round(value)) < EPSILON:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_vector(vec: List[float]) -> str:
    return "(" + ", ".join(format_number(v) for v in vec) + ")"


def format_signed_term(value: float, symbol: str) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {format_number(abs(value))}{symbol}"


def format_general_form(a: float, b: float, c: float) -> str:
    return f"{format_number(a)}x {format_signed_term(b, 'y')} {format_signed_term(c, '')} = 0"


def format_line_equation(m: float, b: float) -> str:
    sign = "+" if b >= 0 else "-"
    return f"y = {format_number(m)}x {sign} {format_number(abs(b))}"


def format_shift(var_name: str, value: float) -> str:
    sign = "-" if value >= 0 else "+"
    return f"({var_name} {sign} {format_number(abs(value))})"


def format_square_term(value: float) -> str:
    if value < 0:
        return f"({format_number(value)})^2"
    return f"{format_number(value)}^2"


def extract_tuples(text: Text) -> List[List[float]]:
    tuples: List[List[float]] = []
    for chunk in TUPLE_RE.findall(text):
        nums = parse_tuple_numbers(chunk)
        if nums:
            tuples.append(nums)
    return tuples


def parse_tuple_numbers(chunk: Text) -> List[float]:
    if "," in chunk:
        parts = [part.strip() for part in chunk.split(",") if part.strip()]
        nums: List[float] = []
        for part in parts:
            match = NUMBER_RE.search(part.replace(",", "."))
            if match:
                nums.append(float(match.group(0).replace(",", ".")))
        return nums
    return parse_numbers(chunk)


def parse_named_value(text: Text, keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        match = re.search(
            rf"\b{re.escape(key)}\s*[:=]\s*(-?\d+(?:[.,]\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def parse_abc(text: Text) -> Dict[Text, float]:
    values: Dict[Text, float] = {}
    for key in ("A", "B", "C"):
        match = re.search(
            rf"\b{key}\s*[:=]\s*(-?\d+(?:[.,]\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            values[key] = float(match.group(1).replace(",", "."))
    return values


def parse_two_points(text: Text) -> Optional[Tuple[List[float], List[float]]]:
    pairs = parse_comma_pairs(text)
    if len(pairs) >= 2:
        return list(pairs[0]), list(pairs[1])
    tuples = extract_tuples(text)
    if len(tuples) >= 2 and len(tuples[0]) >= 2 and len(tuples[1]) >= 2:
        return tuples[0][:2], tuples[1][:2]
    nums = parse_numbers(text)
    if len(nums) >= 4:
        return nums[0:2], nums[2:4]
    return None


def parse_vector(text: Text) -> Optional[List[float]]:
    tuples = extract_tuples(text)
    if tuples and len(tuples[0]) >= 2:
        vec = tuples[0]
        return vec[:3] if len(vec) >= 3 else vec[:2]
    pairs = parse_comma_pairs(text)
    if pairs:
        return list(pairs[0])
    nums = parse_numbers(text)
    if len(nums) >= 3:
        return nums[0:3]
    if len(nums) >= 2:
        return nums[0:2]
    return None


def parse_two_vectors(text: Text) -> Optional[Tuple[List[float], List[float]]]:
    pairs = parse_comma_pairs(text)
    if len(pairs) >= 2:
        v1, v2 = pairs[0], pairs[1]
        return list(v1), list(v2)
    tuples = extract_tuples(text)
    if len(tuples) >= 2:
        v1, v2 = tuples[0], tuples[1]
        dim = min(len(v1), len(v2))
        if dim >= 2:
            return v1[:dim], v2[:dim]
    nums = parse_numbers(text)
    if len(nums) >= 6:
        return nums[0:3], nums[3:6]
    if len(nums) >= 4:
        return nums[0:2], nums[2:4]
    return None


def sanitize_expression(expr: Text) -> Optional[Text]:
    tokens = MATH_RE.findall(expr.replace(",", "."))
    if not tokens:
        return None
    return "".join(tokens)


def normalize_expression(expr: Text) -> Text:
    expr = expr.replace("−", "-").replace("·", "*").replace(" ", "")
    for func in MATH_FUNCS:
        expr = re.sub(rf"{func}\s*([A-Za-z0-9]+)", rf"{func}(\1)", expr)
    expr = insert_implicit_powers(expr)
    return expr


def insert_implicit_powers(expr: Text) -> Text:
    expr = re.sub(r"(?<![A-Za-z])([xyzXYZ])(\d+)", r"\1^\2", expr)
    expr = re.sub(r"\)(\d+)", r")^\1", expr)
    return expr


def extract_expression(text: Text) -> Optional[Text]:
    match = re.search(r"(?:y|f\(x\))\s*=\s*([^\n\r;]+)", text, flags=re.IGNORECASE)
    if match:
        expr = sanitize_expression(match.group(1))
        if expr and re.search(r"[A-Za-z0-9]", expr):
            return normalize_expression(expr)
    candidates = MATH_RE.findall(text)
    if not candidates:
        return None
    preferred = [
        candidate
        for candidate in candidates
        if "x" in candidate.lower() or any(func in candidate.lower() for func in MATH_FUNCS)
    ]
    expr = max(preferred or candidates, key=len)
    expr = sanitize_expression(expr)
    if not expr or not re.search(r"[A-Za-z0-9]", expr):
        return None
    return normalize_expression(expr)


def parse_sympy_expression(expr_text: Text):
    return parse_expr(
        expr_text,
        transformations=TRANSFORMATIONS,
        local_dict={"pi": sp.pi},
        evaluate=True,
    )


def format_expr(expr) -> str:
    return str(expr).replace("**", "^")


def parse_limit_point(text: Text) -> Optional[float]:
    match = re.search(r"x\s*(?:->|→)\s*(-?\d+(?:[.,]\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"x\s*=\s*(-?\d+(?:[.,]\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def try_int(value: float) -> Optional[int]:
    if abs(value - round(value)) < EPSILON:
        return int(round(value))
    return None


def suggest_topics(dispatcher: CollectingDispatcher) -> None:
    dispatcher.utter_message(text=SUGGESTION_TEXT)


def parse_comma_pairs(text: Text) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for x_str, y_str in PAIR_RE.findall(text):
        pairs.append((float(x_str), float(y_str)))
    return pairs


def get_previous_action_name(tracker: Tracker, skip: Optional[Text] = None) -> Optional[Text]:
    skip_names = {"action_listen"}
    if skip:
        skip_names.add(skip)
    for event in reversed(tracker.events):
        if event.get("event") != "action":
            continue
        name = event.get("name")
        if name in skip_names:
            continue
        return name
    return None


class ActionLineEquation(Action):
    def name(self) -> Text:
        return "action_line_equation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        abc = parse_abc(text)
        if len(abc) == 3:
            a_val, b_val, c_val = abc["A"], abc["B"], abc["C"]
            if abs(b_val) < EPSILON:
                if abs(a_val) < EPSILON:
                    dispatcher.utter_message(text="A ба B хоёулаа 0 байна. Шулуун тодорхойгүй.")
                    suggest_topics(dispatcher)
                    return []
                x_value = -c_val / a_val
                dispatcher.utter_message(
                    text=(
                        f"Босоо шулуун: x = {format_number(x_value)}. "
                        f"Ерөнхий хэлбэр: {format_general_form(a_val, b_val, c_val)}"
                    )
                )
                suggest_topics(dispatcher)
                return []
            m_val = -a_val / b_val
            b_intercept = -c_val / b_val
            y_eq = format_line_equation(m_val, b_intercept)
            general = format_general_form(a_val, b_val, c_val)
            dispatcher.utter_message(text=f"{y_eq}. Ерөнхий хэлбэр: {general}")
            suggest_topics(dispatcher)
            return []

        m_val = parse_named_value(text, ("m", "k"))
        b_val = parse_named_value(text, ("b",))
        if m_val is not None and b_val is not None:
            y_eq = format_line_equation(m_val, b_val)
            general = format_general_form(m_val, -1, b_val)
            dispatcher.utter_message(text=f"{y_eq}. Ерөнхий хэлбэр: {general}")
            suggest_topics(dispatcher)
            return []

        points = parse_two_points(text)
        if points:
            (x1, y1), (x2, y2) = points
            if abs(x1 - x2) < EPSILON:
                dispatcher.utter_message(text=f"Босоо шулуун: x = {format_number(x1)}")
                suggest_topics(dispatcher)
                return []
            m_val = (y2 - y1) / (x2 - x1)
            b_intercept = y1 - m_val * x1
            y_eq = format_line_equation(m_val, b_intercept)
            general = format_general_form(m_val, -1, b_intercept)
            dispatcher.utter_message(
                text=(
                    "m = (y2 - y1)/(x2 - x1) = "
                    f"({format_number(y2)} - {format_number(y1)})/"
                    f"({format_number(x2)} - {format_number(x1)}) = {format_number(m_val)}. "
                    f"{y_eq}. Ерөнхий хэлбэр: {general}"
                )
            )
            suggest_topics(dispatcher)
            return []

        dispatcher.utter_message(
            text=(
                "Шулуун: y = m x + b эсвэл A x + B y + C = 0. "
                "Жишээ: (1,2) ба (3,4) цэгээр шулуун ол."
            )
        )
        suggest_topics(dispatcher)
        return []


class ActionDistance(Action):
    def name(self) -> Text:
        return "action_distance"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        points = parse_two_points(text)
        if not points:
            dispatcher.utter_message(
                text="2 цэгийн зай олоход (x1,y1) (x2,y2) хэлбэрээр өгнө үү."
            )
            suggest_topics(dispatcher)
            return []
        (x1, y1), (x2, y2) = points
        dx = x2 - x1
        dy = y2 - y1
        dx2 = dx * dx
        dy2 = dy * dy
        dist = math.sqrt(dx2 + dy2)
        dispatcher.utter_message(
            text=(
                "d = sqrt((x2 - x1)^2 + (y2 - y1)^2) = "
                f"sqrt({format_number(dx2)} + {format_number(dy2)}) = {format_number(dist)}"
            )
        )
        suggest_topics(dispatcher)
        return []


class ActionMidpoint(Action):
    def name(self) -> Text:
        return "action_midpoint"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        points = parse_two_points(text)
        if not points:
            dispatcher.utter_message(
                text="Дунд цэг олоход (x1,y1) (x2,y2) хэлбэрээр өгнө үү."
            )
            suggest_topics(dispatcher)
            return []
        (x1, y1), (x2, y2) = points
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dispatcher.utter_message(
            text=(
                "M = ((x1 + x2)/2, (y1 + y2)/2) = "
                f"({format_number(mx)}, {format_number(my)})"
            )
        )
        suggest_topics(dispatcher)
        return []


class ActionCircleEquation(Action):
    def name(self) -> Text:
        return "action_circle_equation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        tuples = extract_tuples(text)
        center = None
        if tuples and len(tuples[0]) >= 2:
            center = tuples[0][:2]
        r_val = parse_named_value(text, ("r", "radius"))
        nums = parse_numbers(text)
        if center and r_val is None and len(nums) >= 3:
            r_val = nums[-1]
        if not center and len(nums) >= 3:
            center = nums[0:2]
            r_val = nums[2]
        if not center or r_val is None:
            dispatcher.utter_message(
                text="Төв (a,b) ба радиус r өгнө үү. Жишээ: төв (2,3) r=4."
            )
            suggest_topics(dispatcher)
            return []
        a_val, b_val = center
        equation = f"{format_shift('x', a_val)}^2 + {format_shift('y', b_val)}^2 = {format_number(r_val)}^2"
        dispatcher.utter_message(text=f"Тойргийн тэгшитгэл: {equation}")
        suggest_topics(dispatcher)
        return []


class ActionVectorLength(Action):
    def name(self) -> Text:
        return "action_vector_length"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        vec = parse_vector(text)
        if not vec:
            dispatcher.utter_message(text="Векторын бүрэлдэхүүнийг (x,y) хэлбэрээр өгнө үү.")
            suggest_topics(dispatcher)
            return []
        sum_sq = sum(v * v for v in vec)
        length = math.sqrt(sum_sq)
        terms = " + ".join(format_square_term(v) for v in vec)
        dispatcher.utter_message(
            text=f"|v| = sqrt({terms}) = {format_number(length)}"
        )
        suggest_topics(dispatcher)
        return []


class ActionVectorSum(Action):
    def name(self) -> Text:
        return "action_vector_sum"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        vectors = parse_two_vectors(text)
        if not vectors:
            dispatcher.utter_message(text="Хоёр векторыг (x,y) (x,y) хэлбэрээр өгнө үү.")
            suggest_topics(dispatcher)
            return []
        v1, v2 = vectors
        dim = min(len(v1), len(v2))
        result = [v1[i] + v2[i] for i in range(dim)]
        dispatcher.utter_message(
            text=f"a + b = {format_vector(v1[:dim])} + {format_vector(v2[:dim])} = {format_vector(result)}"
        )
        suggest_topics(dispatcher)
        return []


class ActionDotProduct(Action):
    def name(self) -> Text:
        return "action_dot_product"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        vectors = parse_two_vectors(text)
        if not vectors:
            dispatcher.utter_message(text="Хоёр векторыг (x,y) (x,y) хэлбэрээр өгнө үү.")
            suggest_topics(dispatcher)
            return []
        v1, v2 = vectors
        dim = min(len(v1), len(v2))
        terms = " + ".join(
            f"{format_number(v1[i])}*{format_number(v2[i])}" for i in range(dim)
        )
        dot = sum(v1[i] * v2[i] for i in range(dim))
        dispatcher.utter_message(text=f"a dot b = {terms} = {format_number(dot)}")
        suggest_topics(dispatcher)
        return []


class ActionDerivative(Action):
    def name(self) -> Text:
        return "action_derivative"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        expr_text = extract_expression(text)
        if not expr_text:
            dispatcher.utter_message(
                text="Уламжлал олох функцээ өгнө үү. Жишээ: y = x^3 - 2x (эсвэл x3-2x гэж бичиж болно)"
            )
            suggest_topics(dispatcher)
            return []
        try:
            expr = parse_sympy_expression(expr_text)
        except Exception:
            dispatcher.utter_message(text="Функцийн илэрхийлэл танигдсангүй. Жишээ: x^2 + 3x + 1")
            suggest_topics(dispatcher)
            return []
        x = sp.symbols("x")
        deriv = sp.diff(expr, x)
        dispatcher.utter_message(text=f"f(x) = {format_expr(expr)}")
        dispatcher.utter_message(text=f"f'(x) = {format_expr(deriv)}")
        dispatcher.utter_message(
            text="Санамж: (x^n)' = n x^(n-1), (sin x)' = cos x, (cos x)' = -sin x."
        )
        suggest_topics(dispatcher)
        return []


class ActionLimit(Action):
    def name(self) -> Text:
        return "action_limit"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        expr_text = extract_expression(text)
        point = parse_limit_point(text)
        if not expr_text or point is None:
            dispatcher.utter_message(
                text="Предел бодохдоо илэрхийлэл ба x->a утга өгнө үү. Жишээ: limit x->0 sin(x)/x"
            )
            suggest_topics(dispatcher)
            return []
        try:
            expr = parse_sympy_expression(expr_text)
        except Exception:
            dispatcher.utter_message(text="Илэрхийлэл танигдсангүй. Жишээ: (x^2-1)/(x-1)")
            suggest_topics(dispatcher)
            return []
        x = sp.symbols("x")
        try:
            result = sp.limit(expr, x, point)
        except Exception:
            dispatcher.utter_message(text="Пределийг бодох боломжгүй байна. Илэрхийлэлээ шалгана уу.")
            suggest_topics(dispatcher)
            return []
        dispatcher.utter_message(
            text=f"lim x->{format_number(point)} {format_expr(expr)} = {format_expr(result)}"
        )
        suggest_topics(dispatcher)
        return []


class ActionMonotonicity(Action):
    def name(self) -> Text:
        return "action_monotonicity"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        expr_text = extract_expression(text)
        if not expr_text:
            dispatcher.utter_message(text="Квадрат функцээ өгнө үү. Жишээ: y = x^2 - 4x + 1")
            suggest_topics(dispatcher)
            return []
        try:
            expr = parse_sympy_expression(expr_text)
        except Exception:
            dispatcher.utter_message(text="Функцийн илэрхийлэл танигдсангүй.")
            suggest_topics(dispatcher)
            return []
        x = sp.symbols("x")
        try:
            poly = sp.Poly(expr, x)
        except Exception:
            dispatcher.utter_message(text="Полином хэлбэр биш байна. Квадрат функц өгнө үү.")
            suggest_topics(dispatcher)
            return []
        if poly.degree() != 2:
            dispatcher.utter_message(text="Зөвхөн квадрат функц (ax^2+bx+c) дэмжинэ.")
            suggest_topics(dispatcher)
            return []
        coeffs = poly.all_coeffs()
        if any(not coeff.is_number for coeff in coeffs):
            dispatcher.utter_message(text="Тоон коэффициенттэй квадрат функц өгнө үү.")
            suggest_topics(dispatcher)
            return []
        a_val, b_val, c_val = [float(coeff) for coeff in coeffs]
        if abs(a_val) < EPSILON:
            dispatcher.utter_message(text="a=0 тул квадрат функц биш байна.")
            suggest_topics(dispatcher)
            return []
        x0 = -b_val / (2 * a_val)
        y0 = a_val * x0 * x0 + b_val * x0 + c_val
        left = f"(-inf, {format_number(x0)})"
        right = f"({format_number(x0)}, inf)"
        if a_val > 0:
            dispatcher.utter_message(
                text=(
                    f"Өсөх/буурах: буурна {left}, өснө {right}. "
                    f"Минимум: x0 = {format_number(x0)}, f(x0) = {format_number(y0)}."
                )
            )
        else:
            dispatcher.utter_message(
                text=(
                    f"Өсөх/буурах: өснө {left}, буурна {right}. "
                    f"Максимум: x0 = {format_number(x0)}, f(x0) = {format_number(y0)}."
                )
            )
        suggest_topics(dispatcher)
        return []


class ActionProbabilityAdvanced(Action):
    def name(self) -> Text:
        return "action_probability_advanced"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        lower = text.lower()



        n_val = parse_named_value(text, ("n",))
        k_val = parse_named_value(text, ("k",))
        p_val = parse_named_value(text, ("p",))
        has_nkp = n_val is not None and k_val is not None and p_val is not None
        normalized = lower.translate(LOOKALIKE_TRANSLATION)
        is_bernoulli = (
            "bernoulli" in lower
            or "bernulli" in lower
            or "bernouli" in lower
            or "binomial" in lower
            or "binom" in lower
            or "??????" in normalized
            or "?????" in normalized
        )
        if has_nkp or is_bernoulli:
            if not has_nkp:
                dispatcher.utter_message(
                    text="???????? ???? ??? n, k, p ????? ???? ??. ?????: n=5 k=2 p=0.3."
                )
                suggest_topics(dispatcher)
                return []
            n_int = try_int(n_val)
            k_int = try_int(k_val)
            if n_int is None or k_int is None:
                dispatcher.utter_message(text="n ?? k ????? ??? ???? ?????.")
                suggest_topics(dispatcher)
                return []
            if not (0 <= k_int <= n_int):
                dispatcher.utter_message(text="k ?? 0..n ??????? ???? ?????.")
                suggest_topics(dispatcher)
                return []
            prob = math.comb(n_int, k_int) * (p_val ** k_int) * ((1 - p_val) ** (n_int - k_int))
            dispatcher.utter_message(
                text=(
                    f"P(X={k_int}) = C({n_int},{k_int}) p^k (1-p)^(n-k) = {format_number(prob)}"
                )
            )
            suggest_topics(dispatcher)
            return []

        comb_match = re.search(r"[Cc]\((\d+)\s*,\s*(\d+)\)", text)
        if comb_match or any(word in lower for word in ("хослол", "сонгох", "choose")):
            if comb_match:
                n_int = int(comb_match.group(1))
                k_int = int(comb_match.group(2))
            else:
                nums = parse_numbers(text)
                n_int = try_int(nums[0]) if len(nums) >= 2 else None
                k_int = try_int(nums[1]) if len(nums) >= 2 else None
                if n_int is None or k_int is None:
                    n_int = None
            if n_int is not None and k_int is not None:
                if k_int > n_int or n_int < 0 or k_int < 0:
                    dispatcher.utter_message(text="n>=k>=0 байх ёстой.")
                    suggest_topics(dispatcher)
                    return []
                result = math.comb(n_int, k_int)
                dispatcher.utter_message(
                    text=f"C({n_int},{k_int}) = n!/(k!(n-k)!) = {result}"
                )
                suggest_topics(dispatcher)
                return []

        perm_match = re.search(r"[AaPp]\((\d+)\s*,\s*(\d+)\)", text)
        if perm_match or any(word in lower for word in ("байрлал", "пермутац", "permutation")):
            if perm_match:
                n_int = int(perm_match.group(1))
                k_int = int(perm_match.group(2))
            else:
                nums = parse_numbers(text)
                n_int = try_int(nums[0]) if len(nums) >= 2 else None
                k_int = try_int(nums[1]) if len(nums) >= 2 else None
                if n_int is None or k_int is None:
                    n_int = None
            if n_int is not None and k_int is not None:
                if k_int > n_int or n_int < 0 or k_int < 0:
                    dispatcher.utter_message(text="n>=k>=0 байх ёстой.")
                    suggest_topics(dispatcher)
                    return []
                result = math.perm(n_int, k_int)
                dispatcher.utter_message(text=f"P({n_int},{k_int}) = n!/(n-k)! = {result}")
                suggest_topics(dispatcher)
                return []

        fact_match = re.search(r"(\d+)\s*!", text)
        if fact_match or "факториал" in lower:
            if fact_match:
                n_int = int(fact_match.group(1))
            else:
                nums = parse_numbers(text)
                n_int = try_int(nums[0]) if nums else None
            if n_int is not None:
                if n_int < 0:
                    dispatcher.utter_message(text="n нь 0-ээс их байх ёстой.")
                    suggest_topics(dispatcher)
                    return []
                result = math.factorial(n_int)
                dispatcher.utter_message(text=f"{n_int}! = {result}")
                suggest_topics(dispatcher)
                return []

        if "|" in text or "p(a|b)" in lower:
            nums = parse_numbers(text)
            if len(nums) >= 2 and abs(nums[1]) > EPSILON:
                pab, pb = nums[0], nums[1]
                cond = pab / pb
                dispatcher.utter_message(
                    text=f"P(A|B) = P(A∩B)/P(B) = {format_number(cond)}"
                )
                suggest_topics(dispatcher)
                return []

        dispatcher.utter_message(
            text=(
                "Магадлал (ахисан) дэмжлэг: C(n,k), P(n,k), n!, Бернулли (n,k,p), "
                "мөн P(A|B)=P(A∩B)/P(B). Жишээ: C(5,2), A(6,3), 7!, Бернулли: n=5 k=2 p=0.3."
            )
        )
        suggest_topics(dispatcher)
        return []


class ActionHandlePointFollowup(Action):
    def name(self) -> Text:
        return "action_handle_point_followup"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        points = parse_two_points(text)
        previous_action = get_previous_action_name(tracker, skip=self.name())

        if not points:
            dispatcher.utter_message(
                text="Цэгүүдээ (x1,y1) (x2,y2) хэлбэрээр өгнө үү. Жишээ: 6,3 2,4"
            )
            suggest_topics(dispatcher)
            return []

        (x1, y1), (x2, y2) = points

        if previous_action == "action_distance":
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            dispatcher.utter_message(
                text=(
                    "d = sqrt((x2 - x1)^2 + (y2 - y1)^2) = "
                    f"sqrt({format_number(dx * dx)} + {format_number(dy * dy)}) = {format_number(dist)}"
                )
            )
            suggest_topics(dispatcher)
            return []

        if previous_action == "action_midpoint":
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            dispatcher.utter_message(
                text=(
                    "M = ((x1 + x2)/2, (y1 + y2)/2) = "
                    f"({format_number(mx)}, {format_number(my)})"
                )
            )
            suggest_topics(dispatcher)
            return []

        if previous_action == "action_line_equation":
            if abs(x1 - x2) < EPSILON:
                dispatcher.utter_message(text=f"Босоо шулуун: x = {format_number(x1)}")
                suggest_topics(dispatcher)
                return []
            m_val = (y2 - y1) / (x2 - x1)
            b_intercept = y1 - m_val * x1
            y_eq = format_line_equation(m_val, b_intercept)
            general = format_general_form(m_val, -1, b_intercept)
            dispatcher.utter_message(
                text=(
                    "m = (y2 - y1)/(x2 - x1) = "
                    f"({format_number(y2)} - {format_number(y1)})/"
                    f"({format_number(x2)} - {format_number(x1)}) = {format_number(m_val)}. "
                    f"{y_eq}. Ерөнхий хэлбэр: {general}"
                )
            )
            suggest_topics(dispatcher)
            return []

        dispatcher.utter_message(
            text="Ямар бодлого бодох вэ? Шулуун, зай, дунд цэг, тойргийн аль нь вэ?"
        )
        suggest_topics(dispatcher)
        return []


class ActionHandleVectorFollowup(Action):
    def name(self) -> Text:
        return "action_handle_vector_followup"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        previous_action = get_previous_action_name(tracker, skip=self.name())

        if previous_action == "action_vector_length":
            vec = parse_vector(text)
            if not vec:
                dispatcher.utter_message(
                    text="Вектороо (x,y) хэлбэрээр өгнө үү. Жишээ: 3,4"
                )
                suggest_topics(dispatcher)
                return []
            sum_sq = sum(v * v for v in vec)
            length = math.sqrt(sum_sq)
            terms = " + ".join(format_square_term(v) for v in vec)
            dispatcher.utter_message(
                text=f"|v| = sqrt({terms}) = {format_number(length)}"
            )
            suggest_topics(dispatcher)
            return []

        if previous_action in ("action_vector_sum", "action_dot_product"):
            vectors = parse_two_vectors(text)
            if not vectors:
                dispatcher.utter_message(
                    text="Хоёр векторыг (x,y) (x,y) хэлбэрээр өгнө үү. Жишээ: 4,5 -1,-2"
                )
                suggest_topics(dispatcher)
                return []
            v1, v2 = vectors
            dim = min(len(v1), len(v2))
            if previous_action == "action_vector_sum":
                result = [v1[i] + v2[i] for i in range(dim)]
                dispatcher.utter_message(
                    text=(
                        f"a + b = {format_vector(v1[:dim])} + {format_vector(v2[:dim])} "
                        f"= {format_vector(result)}"
                    )
                )
                suggest_topics(dispatcher)
                return []
            terms = " + ".join(
                f"{format_number(v1[i])}*{format_number(v2[i])}" for i in range(dim)
            )
            dot = sum(v1[i] * v2[i] for i in range(dim))
            dispatcher.utter_message(text=f"a dot b = {terms} = {format_number(dot)}")
            suggest_topics(dispatcher)
            return []

        dispatcher.utter_message(
            text="Ямар векторын бодлого бодох вэ? Урт, нийлбэр, скаляр үржвэрийн аль нь вэ?"
        )
        suggest_topics(dispatcher)
        return []
