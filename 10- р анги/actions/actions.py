import math
import re
from typing import Any, Dict, List, Optional, Tuple

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(-?\d+)")
_PI_EXPR_RE = re.compile(r"^[0-9+\-*/().pi]+$")


def _extract_numbers(text: str) -> List[float]:
    if not text:
        return []
    normalized = text.replace(",", ".")
    return [float(value) for value in _NUMBER_RE.findall(normalized)]


def _format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_linear_formula(a: float, b: float) -> str:
    parts = [f"{_format_number(a)}x"]
    if abs(b) > 1e-12:
        sign = "+" if b >= 0 else "-"
        parts.append(f" {sign} {_format_number(abs(b))}")
    return "".join(parts)


def _format_quadratic_formula(a: float, b: float, c: float) -> str:
    parts = [f"{_format_number(a)}x^2"]
    if abs(b) > 1e-12:
        sign = "+" if b >= 0 else "-"
        parts.append(f" {sign} {_format_number(abs(b))}x")
    if abs(c) > 1e-12:
        sign = "+" if c >= 0 else "-"
        parts.append(f" {sign} {_format_number(abs(c))}")
    return "".join(parts)


def _extract_value_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    target = text
    if "=" in text:
        target = text.split("=", 1)[1]
    target = target.strip()
    match = _FRACTION_RE.search(target)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator == 0:
            return None
        return numerator / denominator
    numbers = _extract_numbers(target)
    if numbers:
        return numbers[0]
    return None


def _uses_radians(text: str) -> bool:
    lowered = text.lower()
    return "rad" in lowered or "радиан" in lowered or "pi" in lowered or "π" in text


def _parse_pi_expression(text: str) -> Optional[float]:
    expr = text.replace("π", "pi")
    expr = expr.replace(" ", "")
    expr = expr.replace("радиан", "").replace("rad", "")
    expr = expr.replace("deg", "").replace("градус", "").replace("°", "")
    if not expr:
        return None
    expr = re.sub(r"(\d)pi", r"\1*pi", expr)
    expr = re.sub(r"pi(\d)", r"pi*\1", expr)
    if not _PI_EXPR_RE.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi}))
    except Exception:
        return None


def _parse_angle(text: str) -> Optional[Tuple[float, bool]]:
    if not text:
        return None
    target = text
    if "=" in text:
        target = text.split("=", 1)[1]
    lowered = target.lower()
    if "pi" in lowered or "π" in target:
        radians = _parse_pi_expression(target)
        if radians is None:
            return None
        return radians, True
    value = _extract_value_from_text(target)
    if value is None:
        return None
    if "rad" in lowered or "радиан" in lowered:
        return float(value), True
    return math.radians(float(value)), False


def _unique_angles(values: List[float], period: float, tol: float = 1e-7) -> List[float]:
    normalized: List[float] = []
    for value in values:
        wrapped = value % period
        if not any(abs(wrapped - existing) <= tol for existing in normalized):
            normalized.append(wrapped)
    normalized.sort()
    return normalized


def _format_angle(value: float, use_radians: bool) -> str:
    unit = "рад" if use_radians else "°"
    return f"{_format_number(value)}{unit}"


def _detect_trig_function(text: str) -> Optional[str]:
    lowered = text.lower()
    if "sin" in lowered or "син" in lowered:
        return "sin"
    if "cos" in lowered or "кос" in lowered:
        return "cos"
    if "tan" in lowered or "tg" in lowered or "тан" in lowered:
        return "tan"
    return None


class ActionAskFunctionLinear(Action):
    def name(self) -> str:
        return "action_ask_function_linear"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_function_linear_intro")
        dispatcher.utter_message(response="utter_ask_function_linear")
        return [SlotSet("pending_calc", "function_linear")]


class ActionAskFunctionQuadratic(Action):
    def name(self) -> str:
        return "action_ask_function_quadratic"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_function_quadratic_intro")
        dispatcher.utter_message(response="utter_ask_function_quadratic")
        return [SlotSet("pending_calc", "function_quadratic")]


class ActionAskFunctionRoot(Action):
    def name(self) -> str:
        return "action_ask_function_root"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_function_root_intro")
        dispatcher.utter_message(response="utter_ask_function_root")
        return [SlotSet("pending_calc", "function_root")]


class ActionAskFunctionRational(Action):
    def name(self) -> str:
        return "action_ask_function_rational"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_function_rational_intro")
        dispatcher.utter_message(response="utter_ask_function_rational")
        return [SlotSet("pending_calc", "function_rational")]


class ActionAskFunctionAbsolute(Action):
    def name(self) -> str:
        return "action_ask_function_absolute"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_function_absolute_intro")
        dispatcher.utter_message(response="utter_ask_function_absolute")
        return [SlotSet("pending_calc", "function_absolute")]


class ActionAskFunctionComposition(Action):
    def name(self) -> str:
        return "action_ask_function_composition"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_function_composition_intro")
        dispatcher.utter_message(response="utter_ask_function_composition")
        return [SlotSet("pending_calc", "function_composition")]


class ActionAskTrigUnitCircle(Action):
    def name(self) -> str:
        return "action_ask_trig_unit_circle"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_trig_unit_circle_intro")
        dispatcher.utter_message(response="utter_ask_trig_unit_circle")
        return [SlotSet("pending_calc", "trig_unit_circle")]


class ActionAskTrigEquations(Action):
    def name(self) -> str:
        return "action_ask_trig_equations"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_trig_equations_intro")
        dispatcher.utter_message(response="utter_ask_trig_equations")
        return [SlotSet("pending_calc", "trig_equations")]


class ActionAskExponent(Action):
    def name(self) -> str:
        return "action_ask_exponent"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_exponent_intro")
        dispatcher.utter_message(response="utter_ask_exponent")
        return [SlotSet("pending_calc", "exponent")]


class ActionAskLogarithm(Action):
    def name(self) -> str:
        return "action_ask_logarithm"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_logarithm_intro")
        dispatcher.utter_message(response="utter_ask_logarithm")
        return [SlotSet("pending_calc", "logarithm")]


class ActionAskArithmeticSequence(Action):
    def name(self) -> str:
        return "action_ask_arithmetic_sequence"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_arithmetic_sequence_intro")
        dispatcher.utter_message(response="utter_ask_arithmetic_sequence")
        return [SlotSet("pending_calc", "arithmetic_sequence")]


class ActionAskGeometricSequence(Action):
    def name(self) -> str:
        return "action_ask_geometric_sequence"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_geometric_sequence_intro")
        dispatcher.utter_message(response="utter_ask_geometric_sequence")
        return [SlotSet("pending_calc", "geometric_sequence")]


class ActionHandleNumbers(Action):
    def name(self) -> str:
        return "action_handle_numbers"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        pending = tracker.get_slot("pending_calc")
        text = tracker.latest_message.get("text", "")
        numbers = _extract_numbers(text)

        if not pending:
            dispatcher.utter_message(response="utter_need_topic")
            return []

        if pending == "function_linear":
            if len(numbers) < 3:
                dispatcher.utter_message(response="utter_ask_function_linear")
                return []
            a, b, x_val = numbers[0], numbers[1], numbers[2]
            y_val = a * x_val + b
            formula = _format_linear_formula(a, b)
            message = (
                f"f(x) = {formula}, f({_format_number(x_val)}) = {_format_number(y_val)}"
            )
            if abs(a) > 1e-12:
                x_intercept = -b / a
                message += f", x-огтлол = {_format_number(x_intercept)}"
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "function_quadratic":
            if len(numbers) < 4:
                dispatcher.utter_message(response="utter_ask_function_quadratic")
                return []
            a, b, c, x_val = numbers[0], numbers[1], numbers[2], numbers[3]
            if abs(a) < 1e-12:
                dispatcher.utter_message(text="a=0 үед квадрат функц биш байна.")
                return []
            y_val = a * x_val * x_val + b * x_val + c
            vertex_x = -b / (2 * a)
            vertex_y = a * vertex_x * vertex_x + b * vertex_x + c
            disc = b * b - 4 * a * c
            formula = _format_quadratic_formula(a, b, c)
            message = (
                f"f(x) = {formula}, f({_format_number(x_val)}) = {_format_number(y_val)}. "
                f"Оргил: ({_format_number(vertex_x)}, {_format_number(vertex_y)}), "
                f"тэнхлэг: x = {_format_number(vertex_x)}, D = {_format_number(disc)}."
            )
            if disc > 1e-12:
                sqrt_disc = math.sqrt(disc)
                root1 = (-b - sqrt_disc) / (2 * a)
                root2 = (-b + sqrt_disc) / (2 * a)
                message += (
                    f" Үндэс: x1 = {_format_number(root1)}, "
                    f"x2 = {_format_number(root2)}."
                )
            elif abs(disc) <= 1e-12:
                root = -b / (2 * a)
                message += f" Давхар үндэс: x = {_format_number(root)}."
            else:
                message += " Бодит үндэсгүй."
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "function_root":
            if len(numbers) < 4:
                dispatcher.utter_message(response="utter_ask_function_root")
                return []
            a, h_val, k_val, x_val = numbers[0], numbers[1], numbers[2], numbers[3]
            if x_val < h_val - 1e-12:
                dispatcher.utter_message(text="Домайн: x >= h. Өгөгдсөн x тохирохгүй байна.")
                return []
            y_val = a * math.sqrt(x_val - h_val) + k_val
            if abs(a) < 1e-12:
                range_text = f"y = {_format_number(k_val)}"
            elif a > 0:
                range_text = f"y >= {_format_number(k_val)}"
            else:
                range_text = f"y <= {_format_number(k_val)}"
            message = (
                f"f({_format_number(x_val)}) = {_format_number(y_val)}. "
                f"Домайн: x >= {_format_number(h_val)}, муж: {range_text}."
            )
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "function_rational":
            if len(numbers) < 5:
                dispatcher.utter_message(response="utter_ask_function_rational")
                return []
            a, b, c, d, x_val = numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]
            if abs(c) < 1e-12 and abs(d) < 1e-12:
                dispatcher.utter_message(text="Хуваарь 0 тул функц тодорхойгүй.")
                return []
            denom = c * x_val + d
            if abs(denom) < 1e-12:
                dispatcher.utter_message(text="cx+d=0 тул функц тодорхойгүй байна.")
                return []
            y_val = (a * x_val + b) / denom
            message = f"f({_format_number(x_val)}) = {_format_number(y_val)}."
            if abs(c) > 1e-12:
                x_asym = -d / c
                y_asym = a / c
                message += (
                    f" Домайн: x ≠ {_format_number(x_asym)}, "
                    f"босоо асимптот: x = {_format_number(x_asym)}, "
                    f"хэвтээ асимптот: y = {_format_number(y_asym)}."
                )
            else:
                message += " Домайн: бүх бодит x (d ≠ 0)."
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "function_absolute":
            if len(numbers) < 4:
                dispatcher.utter_message(response="utter_ask_function_absolute")
                return []
            a, h_val, k_val, x_val = numbers[0], numbers[1], numbers[2], numbers[3]
            y_val = a * abs(x_val - h_val) + k_val
            if abs(a) < 1e-12:
                range_text = f"y = {_format_number(k_val)}"
            elif a > 0:
                range_text = f"y >= {_format_number(k_val)}"
            else:
                range_text = f"y <= {_format_number(k_val)}"
            message = (
                f"f({_format_number(x_val)}) = {_format_number(y_val)}, "
                f"оргил: ({_format_number(h_val)}, {_format_number(k_val)}), "
                f"муж: {range_text}."
            )
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "function_composition":
            if len(numbers) < 4:
                dispatcher.utter_message(response="utter_ask_function_composition")
                return []
            a, b, c, d = numbers[0], numbers[1], numbers[2], numbers[3]
            fg_a = a * c
            fg_b = a * d + b
            gf_a = a * c
            gf_b = c * b + d
            fg_formula = _format_linear_formula(fg_a, fg_b)
            gf_formula = _format_linear_formula(gf_a, gf_b)
            message = f"(f∘g)(x) = {fg_formula}, (g∘f)(x) = {gf_formula}"
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "trig_unit_circle":
            parsed = _parse_angle(text)
            if not parsed:
                dispatcher.utter_message(response="utter_ask_trig_unit_circle")
                return []
            radians, use_radians = parsed
            degrees = math.degrees(radians)
            sin_val = math.sin(radians)
            cos_val = math.cos(radians)
            if abs(sin_val) < 1e-12:
                sin_val = 0.0
            if abs(cos_val) < 1e-12:
                cos_val = 0.0
            if abs(cos_val) < 1e-12:
                tan_text = "тодорхойгүй"
            else:
                tan_val = sin_val / cos_val
                tan_text = _format_number(tan_val)
            if use_radians:
                angle_line = (
                    f"Өнцөг: {_format_number(radians)} рад (~{_format_number(degrees)}°)"
                )
            else:
                angle_line = (
                    f"Өнцөг: {_format_number(degrees)}° (~{_format_number(radians)} рад)"
                )
            message = (
                f"{angle_line}\n"
                f"sin = {_format_number(sin_val)}, cos = {_format_number(cos_val)}, tan = {tan_text}"
            )
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "trig_equations":
            func = _detect_trig_function(text)
            if not func:
                dispatcher.utter_message(response="utter_ask_trig_equations")
                return []
            value = _extract_value_from_text(text)
            if value is None:
                dispatcher.utter_message(response="utter_ask_trig_equations")
                return []
            use_radians = _uses_radians(text)
            if func in {"sin", "cos"} and abs(value) > 1 + 1e-9:
                dispatcher.utter_message(text="|a| > 1 тул бодит шийдгүй.")
                return []

            if use_radians:
                period = 2 * math.pi
                if func == "sin":
                    alpha = math.asin(value)
                    base_solutions = [alpha, math.pi - alpha]
                    general = (
                        f"x = {_format_number(alpha)} + 2πk, "
                        f"x = {_format_number(math.pi - alpha)} + 2πk"
                    )
                elif func == "cos":
                    alpha = math.acos(value)
                    base_solutions = [alpha, -alpha]
                    general = (
                        f"x = {_format_number(alpha)} + 2πk, "
                        f"x = {_format_number(-alpha)} + 2πk"
                    )
                else:
                    alpha = math.atan(value)
                    base_solutions = [alpha, alpha + math.pi]
                    general = f"x = {_format_number(alpha)} + πk"
                base_list = _unique_angles(base_solutions, period)
                base_text = ", ".join(_format_angle(val, True) for val in base_list)
                message = (
                    f"0 ≤ x < 2π: x = {base_text}.\n"
                    f"Ерөнхий: {general}."
                )
            else:
                period = 360.0
                if func == "sin":
                    alpha = math.degrees(math.asin(value))
                    base_solutions = [alpha, 180 - alpha]
                    general = (
                        f"x = {_format_number(alpha)}° + 360°k, "
                        f"x = {_format_number(180 - alpha)}° + 360°k"
                    )
                elif func == "cos":
                    alpha = math.degrees(math.acos(value))
                    base_solutions = [alpha, -alpha]
                    general = (
                        f"x = {_format_number(alpha)}° + 360°k, "
                        f"x = {_format_number(-alpha)}° + 360°k"
                    )
                else:
                    alpha = math.degrees(math.atan(value))
                    base_solutions = [alpha, alpha + 180]
                    general = f"x = {_format_number(alpha)}° + 180°k"
                base_list = _unique_angles(base_solutions, period)
                base_text = ", ".join(_format_angle(val, False) for val in base_list)
                message = (
                    f"0° ≤ x < 360°: x = {base_text}.\n"
                    f"Ерөнхий: {general}."
                )
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "exponent":
            if len(numbers) < 2:
                dispatcher.utter_message(response="utter_ask_exponent")
                return []
            base, exponent = numbers[0], numbers[1]
            if base <= 0:
                dispatcher.utter_message(text="Суурь a>0 байх ёстой.")
                return []
            result = base ** exponent
            trend = "өсөлт" if base > 1 else "бууралт" if 0 < base < 1 else "тогтмол"
            message = (
                f"{_format_number(base)}^{_format_number(exponent)} = {_format_number(result)} "
                f"({trend})."
            )
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "logarithm":
            if len(numbers) < 2:
                dispatcher.utter_message(response="utter_ask_logarithm")
                return []
            base, value = numbers[0], numbers[1]
            if base <= 0 or abs(base - 1) < 1e-12 or value <= 0:
                dispatcher.utter_message(text="a>0, a≠1, b>0 нөхцөл шаардлагатай.")
                return []
            result = math.log(value, base)
            dispatcher.utter_message(
                text=f"log_{_format_number(base)}({_format_number(value)}) = {_format_number(result)}"
            )
            return [SlotSet("pending_calc", None)]

        if pending == "arithmetic_sequence":
            if len(numbers) < 3:
                dispatcher.utter_message(response="utter_ask_arithmetic_sequence")
                return []
            a1, d, n_val = numbers[0], numbers[1], numbers[2]
            n_int = int(round(n_val))
            if n_int <= 0 or abs(n_val - n_int) > 1e-9:
                dispatcher.utter_message(text="n нь эерэг бүхэл тоо байх ёстой.")
                return []
            a_n = a1 + (n_int - 1) * d
            s_n = n_int / 2 * (2 * a1 + (n_int - 1) * d)
            message = (
                f"a_{n_int} = {_format_number(a_n)}, "
                f"S_{n_int} = {_format_number(s_n)}"
            )
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        if pending == "geometric_sequence":
            if len(numbers) < 3:
                dispatcher.utter_message(response="utter_ask_geometric_sequence")
                return []
            a1, q, n_val = numbers[0], numbers[1], numbers[2]
            n_int = int(round(n_val))
            if n_int <= 0 or abs(n_val - n_int) > 1e-9:
                dispatcher.utter_message(text="n нь эерэг бүхэл тоо байх ёстой.")
                return []
            a_n = a1 * (q ** (n_int - 1))
            if abs(q - 1) < 1e-12:
                s_n = a1 * n_int
            else:
                s_n = a1 * ((q ** n_int) - 1) / (q - 1)
            message = (
                f"a_{n_int} = {_format_number(a_n)}, "
                f"S_{n_int} = {_format_number(s_n)}"
            )
            dispatcher.utter_message(text=message)
            return [SlotSet("pending_calc", None)]

        dispatcher.utter_message(response="utter_need_topic")
        return []
