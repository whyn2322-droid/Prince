from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sympy import I, factor, expand, simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.simplify.radsimp import radsimp

EPSILON = 1e-9
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def parse_numbers(text: Text) -> List[float]:
    return [float(n.replace(",", ".")) for n in NUMBER_RE.findall(text)]


def format_number(value: float) -> str:
    if abs(value - round(value)) < EPSILON:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_fraction(numerator: int, denominator: int) -> str:
    gcd_value = math.gcd(numerator, denominator)
    numerator //= gcd_value
    denominator //= gcd_value
    return f"{numerator}/{denominator} (~{numerator / denominator:.4f})"


def parse_named_coefficients(text: Text) -> Dict[Text, float]:
    coeffs: Dict[Text, float] = {}
    for name in ("a", "b", "c"):
        match = re.search(
            rf"\b{name}\s*[:=]\s*(-?\d+(?:[.,]\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            coeffs[name] = float(match.group(1).replace(",", "."))
    return coeffs


def parse_quadratic_coeffs(text: Text) -> Optional[Tuple[float, float, float]]:
    named = parse_named_coefficients(text)
    if all(name in named for name in ("a", "b", "c")):
        return named["a"], named["b"], named["c"]
    numbers = parse_numbers(text)
    if len(numbers) >= 3:
        return numbers[0], numbers[1], numbers[2]
    return None


def extract_expression(text: Text, allow_sqrt: bool = False) -> Optional[Text]:
    allowed = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^().")
    if allow_sqrt:
        allowed.add("√")
    cleaned = "".join(ch if ch in allowed else " " for ch in text)
    cleaned = cleaned.replace(" ", "")
    return cleaned or None


def normalize_sqrt_text(text: Text) -> Text:
    text = text.replace("√", "sqrt")
    text = re.sub(r"sqrt\s*(?!\()([0-9.]+)", r"sqrt(\1)", text)
    text = re.sub(r"sqrt\s*(?!\()([A-Za-z]+)", r"sqrt(\1)", text)
    return text


def parse_sympy_expression(text: Text):
    normalized = text.replace("^", "**")
    return parse_expr(normalized, transformations=TRANSFORMATIONS, evaluate=True)


def format_sympy_expr(expr) -> str:
    return str(expr).replace("**", "^")


def parse_scale_ratio(text: Text) -> Optional[Tuple[float, float]]:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)", text)
    if not match:
        return None
    left = float(match.group(1).replace(",", "."))
    right = float(match.group(2).replace(",", "."))
    return left, right


def parse_lengths_with_units(text: Text) -> List[Tuple[float, str, float]]:
    unit_map = {
        "см": 1.0,
        "cm": 1.0,
        "м": 100.0,
        "метр": 100.0,
        "км": 100000.0,
    }
    results: List[Tuple[float, str, float]] = []
    for value_str, unit in re.findall(
        r"(-?\d+(?:[.,]\d+)?)\s*(см|cm|м|метр|км)", text, flags=re.IGNORECASE
    ):
        value = float(value_str.replace(",", "."))
        unit_lower = unit.lower()
        cm_value = value * unit_map[unit_lower]
        results.append((value, unit_lower, cm_value))
    return results


def format_pi_value(coef: float) -> str:
    coef_text = format_number(coef)
    if coef_text == "0":
        return "0"
    if coef_text == "1":
        return "pi"
    return f"{coef_text}*pi"


class ActionSolveQuadratic(Action):
    def name(self) -> Text:
        return "action_solve_quadratic"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        coeffs = parse_quadratic_coeffs(text)
        if coeffs is None:
            dispatcher.utter_message(
                text="a, b, c коэффициентээ өгнө үү. Жишээ: квадрат тэгшитгэл бод 1 -3 2"
            )
            return []
        a, b, c = coeffs

        if abs(a) < EPSILON:
            if abs(b) < EPSILON:
                if abs(c) < EPSILON:
                    dispatcher.utter_message(
                        text="0 = 0 тул хязгааргүй олон шийдтэй."
                    )
                else:
                    dispatcher.utter_message(
                        text="0x + c = 0 хэлбэр тул шийдгүй."
                    )
                return []
            x = -c / b
            dispatcher.utter_message(
                text=f"a=0 тул шугаман тэгшитгэл. x = {format_number(x)}"
            )
            return []

        d = b * b - 4 * a * c
        dispatcher.utter_message(text="Томьёо: x = (-b +/- sqrt(D)) / (2a)")
        dispatcher.utter_message(text=f"D = b^2 - 4ac = {format_number(d)}")

        if d > EPSILON:
            sqrt_d = math.sqrt(d)
            x1 = (-b + sqrt_d) / (2 * a)
            x2 = (-b - sqrt_d) / (2 * a)
            dispatcher.utter_message(
                text=(
                    "Хоёр бодит шийд: "
                    f"x1 = {format_number(x1)}, x2 = {format_number(x2)}"
                )
            )
        elif abs(d) <= EPSILON:
            x = -b / (2 * a)
            dispatcher.utter_message(text=f"Давхар шийд: x = {format_number(x)}")
        else:
            dispatcher.utter_message(text="D < 0 тул бодит шийдгүй.")
            dispatcher.utter_message(
                text="Хэрэв комплекс авч үзвэл: x = (-b +/- i*sqrt(|D|)) / (2a)"
            )
        return []


class ActionDiscriminant(Action):
    def name(self) -> Text:
        return "action_discriminant"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()
        coeffs = parse_quadratic_coeffs(text)
        if coeffs is None:
            if "томьёо" in text_lower or "гэж юу" in text_lower:
                dispatcher.utter_message(
                    text="Дискриминант: D = b^2 - 4ac. D>0 бол 2 бодит шийд, D=0 бол 1 давхар шийд, D<0 бол бодит шийдгүй."
                )
                return []
            dispatcher.utter_message(
                text="a, b, c-гээ өгнө үү. Жишээ: дискриминант ол 1 -3 2"
            )
            return []
        a, b, c = coeffs
        d = b * b - 4 * a * c
        if abs(a) < EPSILON:
            dispatcher.utter_message(
                text="a=0 тул квадрат тэгшитгэл биш боловч D = b^2 - 4ac томьёогоор тооцлоо."
            )
        dispatcher.utter_message(text=f"D = b^2 - 4ac = {format_number(d)}")
        if d > EPSILON:
            dispatcher.utter_message(text="D > 0 тул 2 бодит шийдтэй.")
        elif abs(d) <= EPSILON:
            dispatcher.utter_message(text="D = 0 тул 1 давхар шийдтэй.")
        else:
            dispatcher.utter_message(text="D < 0 тул бодит шийдгүй.")
        return []


class ActionFactorOrExpand(Action):
    def name(self) -> Text:
        return "action_factor_or_expand"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()
        expand_mode = any(word in text_lower for word in ("дэлг", "expand"))

        expr_text = extract_expression(text)
        if not expr_text:
            dispatcher.utter_message(
                text="Илэрхийлэл өгнө үү. Жишээ: x^2-5x+6 -г факторчил, эсвэл (x+3)(x-2) -г дэлгэж өг"
            )
            return []
        try:
            expr = parse_sympy_expression(expr_text)
        except Exception:
            dispatcher.utter_message(text="Илэрхийлэл танигдсангүй. Жишээ: 2x^2+5x+3")
            return []

        if expand_mode:
            result = expand(expr)
            dispatcher.utter_message(
                text=f"Дэлгэсэн хэлбэр: {format_sympy_expr(result)}"
            )
        else:
            result = factor(expr)
            dispatcher.utter_message(
                text=f"Факторчилсон хэлбэр: {format_sympy_expr(result)}"
            )
        return []


class ActionSimilarityScale(Action):
    def name(self) -> Text:
        return "action_similarity_scale"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()

        if "периметр" in text_lower and ("харьцаа" in text_lower or "харьцаатай" in text_lower):
            numbers = parse_numbers(text_lower)
            if len(numbers) >= 2:
                a, b = numbers[-2], numbers[-1]
                dispatcher.utter_message(
                    text=f"Периметрүүдийн харьцаа талуудын адил: {format_number(a)}:{format_number(b)}"
                )
                return []

        if "талбай" in text_lower and "масштаб" in text_lower:
            numbers = parse_numbers(text_lower)
            if numbers:
                area_factor = numbers[0]
                scale = math.sqrt(area_factor)
                dispatcher.utter_message(
                    text=f"Талбай k^2-оор өснө. k = sqrt({format_number(area_factor)}) = {format_number(scale)}"
                )
                return []

        if "урт" in text_lower and "эзэлхүүн" in text_lower:
            numbers = parse_numbers(text_lower)
            if numbers:
                length_factor = numbers[0]
                volume_factor = length_factor ** 3
                dispatcher.utter_message(
                    text=f"Эзэлхүүн k^3-оор өөрчлөгдөнө: {format_number(volume_factor)} дахин."
                )
                return []

        if "томруул" in text_lower and "масштаб" in text_lower:
            numbers = parse_numbers(text_lower)
            if numbers:
                factor = numbers[0]
                dispatcher.utter_message(
                    text=f"Масштаб = {format_number(factor)}:1 (эсвэл 1:{format_number(1 / factor)})"
                )
                return []

        if "масштаб" in text_lower and ("ол" in text_lower or "хэд" in text_lower) and "1:" not in text_lower:
            lengths = parse_lengths_with_units(text_lower)
            if len(lengths) >= 2:
                map_cm = lengths[0][2]
                real_cm = lengths[1][2]
                if map_cm > EPSILON:
                    scale = real_cm / map_cm
                    dispatcher.utter_message(
                        text=f"Масштаб = 1:{format_number(scale)}"
                    )
                    return []

        ratio = parse_scale_ratio(text_lower)
        lengths = parse_lengths_with_units(text_lower)
        if ratio and lengths:
            left, right = ratio
            if abs(left) < EPSILON:
                dispatcher.utter_message(text="Масштабын зүүн тал 0 байж болохгүй.")
                return []
            map_cm = lengths[0][2]
            real_cm = map_cm * (right / left)
            if "км" in text_lower:
                km = real_cm / 100000.0
                dispatcher.utter_message(text=f"Бодит зай: {format_number(km)} км")
                return []
            if "метр" in text_lower:
                meters = real_cm / 100.0
                dispatcher.utter_message(text=f"Бодит зай: {format_number(meters)} м")
                return []
            dispatcher.utter_message(
                text=f"Бодит зай: {format_number(real_cm)} см = {format_number(real_cm / 100.0)} м"
            )
            return []

        if "масштаб" in text_lower:
            match_ab = re.search(r"ab\s*=\s*(-?\d+(?:[.,]\d+)?)", text_lower)
            match_apb = re.search(
                r"a['’]b['’]\s*=\s*(-?\d+(?:[.,]\d+)?)", text_lower
            )
            if match_ab and match_apb:
                ab = float(match_ab.group(1).replace(",", "."))
                apb = float(match_apb.group(1).replace(",", "."))
                if abs(ab) > EPSILON:
                    scale = apb / ab
                    dispatcher.utter_message(
                        text=f"Масштаб k = {format_number(scale)}"
                    )
                    return []

        dispatcher.utter_message(
            text="Төсөө/масштаб: уртын харьцаа k, талбай k^2, эзэлхүүн k^3. Жишээ: масштаб 1:500 дээр 4см нь бодит хэд вэ?"
        )
        return []


class ActionProbability(Action):
    def name(self) -> Text:
        return "action_probability"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()

        if "шоо" in text_lower:
            two_times = ("2" in text_lower or "хоёр" in text_lower) and "удаа" in text_lower
            if two_times and "6" in text_lower:
                dispatcher.utter_message(text=f"P = {format_fraction(1, 36)}")
                return []
            if "6" in text_lower:
                dispatcher.utter_message(text=f"P = {format_fraction(1, 6)}")
                return []

        if "зос" in text_lower and ("3" in text_lower or "гурав" in text_lower):
            if ("2" in text_lower or "хоёр" in text_lower) and "сүлд" in text_lower:
                dispatcher.utter_message(text=f"P = {format_fraction(3, 8)}")
                return []

        if "улаан" in text_lower and "цэнхэр" in text_lower:
            red_match = re.search(r"(\d+)\s*улаан", text_lower)
            blue_match = re.search(r"(\d+)\s*цэнхэр", text_lower)
            red = int(red_match.group(1)) if red_match else 3
            blue = int(blue_match.group(1)) if blue_match else 2
            total = red + blue
            if "буцаахгүй" in text_lower and "цэнхэр" in text_lower:
                dispatcher.utter_message(
                    text=f"P = {format_fraction(blue, total)} (өөрчлөхгүй гэж үзэв)"
                )
                return []
            if "улаан" in text_lower and "магадлал" in text_lower and "цэнхэр гарах" not in text_lower:
                dispatcher.utter_message(text=f"P = {format_fraction(red, total)}")
                return []
            if "цэнхэр" in text_lower and "магадлал" in text_lower:
                dispatcher.utter_message(text=f"P = {format_fraction(blue, total)}")
                return []

        dispatcher.utter_message(
            text="Магадлал: P(A)=таатай/нийт. Жишээ: шоо нэг шидэхэд 6 буух магадлал хэд вэ?"
        )
        return []


class ActionStatistics(Action):
    def name(self) -> Text:
        return "action_statistics"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()
        numbers = parse_numbers(text)
        if not numbers:
            if "суурь" in text_lower or "үндэс" in text_lower:
                dispatcher.utter_message(
                    text="Статистик: дундаж, медиан, мод, хүрээ, дисперс зэрэг үзүүлэлтүүдтэй."
                )
                return []
            dispatcher.utter_message(
                text="Өгөгдлөө өгнө үү. Жишээ: дундаж ол 2 4 6 8 10"
            )
            return []

        if "дундаж" in text_lower:
            mean = sum(numbers) / len(numbers)
            dispatcher.utter_message(text=f"Дундаж = {format_number(mean)}")
            return []

        if "медиан" in text_lower:
            nums = sorted(numbers)
            n = len(nums)
            if n % 2 == 1:
                median = nums[n // 2]
            else:
                median = (nums[n // 2 - 1] + nums[n // 2]) / 2
            dispatcher.utter_message(text=f"Медиан = {format_number(median)}")
            return []

        if "мод" in text_lower:
            freq: Dict[float, int] = {}
            for num in numbers:
                freq[num] = freq.get(num, 0) + 1
            max_count = max(freq.values())
            modes = [num for num, count in freq.items() if count == max_count]
            mode_text = ", ".join(format_number(num) for num in sorted(modes))
            dispatcher.utter_message(text=f"Мод = {mode_text}")
            return []

        if "хүрээ" in text_lower or "range" in text_lower:
            data_range = max(numbers) - min(numbers)
            dispatcher.utter_message(text=f"Хүрээ = {format_number(data_range)}")
            return []

        if "дисперс" in text_lower:
            mean = sum(numbers) / len(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            dispatcher.utter_message(
                text=f"Дисперс (суурь, n-т хуваасан) = {format_number(variance)}"
            )
            return []

        dispatcher.utter_message(
            text="Статистикийн төрөл танигдсангүй (дундаж/медиан/мод/хүрээ/дисперс)."
        )
        return []


class ActionTrigonometry(Action):
    def name(self) -> Text:
        return "action_trigonometry"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()

        if "sin^2" in text_lower and "cos^2" in text_lower:
            dispatcher.utter_message(
                text="sin^2(x) + cos^2(x) = 1 нь нэгж тойргийн үндсэн адилтгал."
            )
            return []

        if "радиан" in text_lower and "градус" in text_lower:
            if "pi" in text_lower or "π" in text:
                match = re.search(r"(\d+)?\s*(pi|π)\s*/\s*(\d+)", text_lower)
                if match:
                    numerator = int(match.group(1) or "1")
                    denominator = int(match.group(3))
                    degrees = 180 * numerator / denominator
                    if numerator == 1:
                        rad_text = f"pi/{denominator}"
                    else:
                        rad_text = f"{numerator}*pi/{denominator}"
                    dispatcher.utter_message(
                        text=f"{rad_text} радиан = {format_number(degrees)} градус"
                    )
                    return []
            match = re.search(r"(-?\d+(?:[.,]\d+)?)\s*градус", text_lower)
            if match:
                deg = float(match.group(1).replace(",", "."))
                exact = {
                    30: "pi/6",
                    45: "pi/4",
                    60: "pi/3",
                    90: "pi/2",
                    180: "pi",
                    135: "3*pi/4",
                    150: "5*pi/6",
                }
                if deg in exact:
                    dispatcher.utter_message(
                        text=f"{format_number(deg)} градус = {exact[deg]} радиан"
                    )
                else:
                    rad = deg * math.pi / 180
                    dispatcher.utter_message(
                        text=f"{format_number(deg)} градус = {format_number(rad)} радиан"
                    )
                return []

        sin_values = {0: "0", 30: "1/2", 90: "1", 150: "1/2"}
        cos_values = {0: "1", 60: "1/2", 180: "-1"}
        tan_values = {45: "1", 135: "-1"}

        if "sin" in text_lower:
            angles = [int(a) for a in re.findall(r"sin\s*\(?\s*(\d+)", text_lower)]
            if "pi/6" in text_lower or "π/6" in text:
                dispatcher.utter_message(text="sin(pi/6) = 1/2")
                return []
            if angles:
                parts = []
                for angle in angles:
                    value = sin_values.get(angle)
                    if value is None:
                        value = format_number(math.sin(math.radians(angle)))
                    parts.append(f"sin {angle} = {value}")
                dispatcher.utter_message(text=", ".join(parts))
                return []

        if "cos" in text_lower:
            angles = [int(a) for a in re.findall(r"cos\s*\(?\s*(\d+)", text_lower)]
            if "pi/3" in text_lower or "π/3" in text:
                dispatcher.utter_message(text="cos(pi/3) = 1/2")
                return []
            if angles:
                parts = []
                for angle in angles:
                    value = cos_values.get(angle)
                    if value is None:
                        value = format_number(math.cos(math.radians(angle)))
                    parts.append(f"cos {angle} = {value}")
                dispatcher.utter_message(text=", ".join(parts))
                return []

        if "tan" in text_lower:
            angles = [int(a) for a in re.findall(r"tan\s*\(?\s*(\d+)", text_lower)]
            if angles:
                parts = []
                for angle in angles:
                    if angle == 90:
                        value = "тодорхойгүй"
                    else:
                        value = tan_values.get(angle)
                        if value is None:
                            value = format_number(math.tan(math.radians(angle)))
                    parts.append(f"tan {angle} = {value}")
                dispatcher.utter_message(text=", ".join(parts))
                return []

        dispatcher.utter_message(
            text="Суурь томьёо: sin(theta)=эсрэг/гипотенуз, cos(theta)=залгаа/гипотенуз, tan(theta)=эсрэг/залгаа."
        )
        return []


class ActionSquareRoots(Action):
    def name(self) -> Text:
        return "action_square_roots"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()

        expr_text = extract_expression(text, allow_sqrt=True)
        if not expr_text:
            dispatcher.utter_message(
                text="Квадрат язгуур: a^2 = a*a, sqrt(a^2)=|a|. Жишээ: √50, √72, 1/√5"
            )
            return []
        expr_text = normalize_sqrt_text(expr_text)

        try:
            expr = parse_sympy_expression(expr_text)
        except Exception:
            dispatcher.utter_message(text="Язгуурын илэрхийлэл танигдсангүй.")
            return []

        if "язгуургүй" in text_lower or "rationalize" in text_lower:
            result = radsimp(expr)
            dispatcher.utter_message(text=f"Хариу: {format_sympy_expr(result)}")
            return []

        result = simplify(expr)
        if result.has(I):
            dispatcher.utter_message(
                text="Бодит тоонд боломжгүй. Комплекс тоонд: "
                f"{format_sympy_expr(result)}"
            )
            return []
        dispatcher.utter_message(text=f"Хариу: {format_sympy_expr(result)}")
        return []


class ActionCone(Action):
    def name(self) -> Text:
        return "action_cone"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()

        def extract_value(patterns: List[str]) -> Optional[float]:
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return float(match.group(1).replace(",", "."))
            return None

        r_value = extract_value([r"\br\s*[:=]\s*(-?\d+(?:[.,]\d+)?)", r"радиус\s*(-?\d+(?:[.,]\d+)?)"])
        h_value = extract_value([r"\bh\s*[:=]\s*(-?\d+(?:[.,]\d+)?)", r"өндөр\s*(-?\d+(?:[.,]\d+)?)"])
        l_value = extract_value([r"\bl\s*[:=]\s*(-?\d+(?:[.,]\d+)?)", r"ташуу\s*[-\w]*\s*(-?\d+(?:[.,]\d+)?)"])
        d_value = extract_value([r"диаметр\s*(-?\d+(?:[.,]\d+)?)", r"\bd\s*[:=]\s*(-?\d+(?:[.,]\d+)?)"])

        if "бөмбөрцөг" in text_lower and "конус" in text_lower and not any(
            word in text_lower for word in ("r=", "h=", "l=", "диаметр")
        ):
            dispatcher.utter_message(
                text=(
                    "Конус: V = (1/3)*pi*r^2*h, l = sqrt(r^2 + h^2), "
                    "Sx = pi*r*l, S = pi*r^2 + pi*r*l. "
                    "Бөмбөрцөг: S = 4*pi*r^2, V = (4/3)*pi*r^3."
                )
            )
            return []

        if "байгуулагч" in text_lower or "ташуу" in text_lower:
            if r_value is not None and h_value is not None:
                l_calc = math.sqrt(r_value ** 2 + h_value ** 2)
                dispatcher.utter_message(
                    text=f"l = sqrt(r^2 + h^2) = {format_number(l_calc)}"
                )
                return []
            dispatcher.utter_message(
                text="l = sqrt(r^2 + h^2) томьёогоор олно. r, h-ийг өгнө үү."
            )
            return []

        if "хажуу" in text_lower:
            if r_value is not None and l_value is not None:
                coef = r_value * l_value
                dispatcher.utter_message(
                    text=f"Sx = pi*r*l = {format_pi_value(coef)} (~{format_number(coef * math.pi)})"
                )
                return []
            dispatcher.utter_message(
                text="Хажуу гадаргуу: Sx = pi*r*l. r, l-ийг өгнө үү."
            )
            return []

        if "нийт" in text_lower:
            if r_value is not None and l_value is not None:
                coef = r_value ** 2 + r_value * l_value
                dispatcher.utter_message(
                    text=f"S = pi*(r^2 + r*l) = {format_pi_value(coef)} (~{format_number(coef * math.pi)})"
                )
                return []
            dispatcher.utter_message(
                text="Нийт гадаргуу: S = pi*r^2 + pi*r*l. r, l-ийг өгнө үү."
            )
            return []

        if "эзэлхүүн" in text_lower or "v" in text_lower:
            if r_value is None and d_value is not None:
                r_value = d_value / 2
            if r_value is not None and h_value is not None:
                coef = (r_value ** 2 * h_value) / 3
                dispatcher.utter_message(
                    text=f"V = (1/3)*pi*r^2*h = {format_pi_value(coef)} (~{format_number(coef * math.pi)})"
                )
                return []
            dispatcher.utter_message(
                text="Эзэлхүүн: V = (1/3)*pi*r^2*h. r, h (эсвэл диаметр, өндөр)-ийг өгнө үү."
            )
            return []

        dispatcher.utter_message(
            text="Конусын томьёо: V=(1/3)*pi*r^2*h, l=sqrt(r^2+h^2), Sx=pi*r*l, S=pi*r^2+pi*r*l."
        )
        return []


class ActionSphere(Action):
    def name(self) -> Text:
        return "action_sphere"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "")
        text_lower = text.lower()

        def extract_value(patterns: List[str]) -> Optional[float]:
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return float(match.group(1).replace(",", "."))
            return None

        r_value = extract_value([r"\br\s*[:=]\s*(-?\d+(?:[.,]\d+)?)", r"радиус\s*(-?\d+(?:[.,]\d+)?)"])
        d_value = extract_value([r"диаметр\s*(-?\d+(?:[.,]\d+)?)", r"\bd\s*[:=]\s*(-?\d+(?:[.,]\d+)?)"])

        v_match = re.search(r"v\s*=\s*(-?\d+(?:[.,]\d+)?)\s*(pi|π)", text_lower)
        if v_match and ("r ол" in text_lower or "радиус" in text_lower):
            coef = float(v_match.group(1).replace(",", "."))
            r_calc = (3 * coef / 4) ** (1 / 3)
            dispatcher.utter_message(
                text=f"V = (4/3)*pi*r^3 => r = {format_number(r_calc)}"
            )
            return []

        if r_value is None and d_value is not None:
            r_value = d_value / 2

        if "эзэлхүүн" in text_lower or "v" in text_lower:
            if r_value is not None:
                coef = (4 * r_value ** 3) / 3
                dispatcher.utter_message(
                    text=f"V = (4/3)*pi*r^3 = {format_pi_value(coef)} (~{format_number(coef * math.pi)})"
                )
                return []
            dispatcher.utter_message(
                text="Эзэлхүүн: V = (4/3)*pi*r^3. r (эсвэл диаметр)-ийг өгнө үү."
            )
            return []

        if "гадаргуу" in text_lower:
            if r_value is not None:
                coef = 4 * r_value ** 2
                dispatcher.utter_message(
                    text=f"S = 4*pi*r^2 = {format_pi_value(coef)} (~{format_number(coef * math.pi)})"
                )
                return []
            dispatcher.utter_message(text="Гадаргуу: S = 4*pi*r^2. r-ийг өгнө үү.")
            return []

        dispatcher.utter_message(
            text="Бөмбөрцөг: S = 4*pi*r^2, V = (4/3)*pi*r^3."
        )
        return []
