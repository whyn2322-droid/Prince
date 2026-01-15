from typing import Any, Dict, List, Text, Optional
import math
import re

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import AllSlotsReset

POSITIVE_NUMBER_MESSAGE = "Эерэг тоо оруулна уу. (ж: 5 эсвэл 3.5)"
NUMBER_MESSAGE = "Зөв тоо оруулна уу. (ж: -2, 0, 3.5)"


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


def _positive(x: Optional[float]) -> bool:
    return x is not None and x > 0


def _fmt_num(value: float) -> str:
    return f"{value:g}"


def _simplify_ratio(a: float, b: float) -> Optional[str]:
    if not a.is_integer() or not b.is_integer():
        return None
    a_i = int(abs(a))
    b_i = int(abs(b))
    if a_i == 0 or b_i == 0:
        return None
    g = math.gcd(a_i, b_i)
    return f"{a_i // g}:{b_i // g}"


def _get_slot_float(tracker: Tracker, slot_name: Text) -> Optional[float]:
    value = tracker.get_slot(slot_name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_positive_slots(
    dispatcher: CollectingDispatcher, tracker: Tracker, slot_names: List[Text]
) -> Optional[List[float]]:
    values: List[float] = []
    for slot_name in slot_names:
        num = _get_slot_float(tracker, slot_name)
        if not _positive(num):
            dispatcher.utter_message(text=POSITIVE_NUMBER_MESSAGE)
            return None
        values.append(num)
    return values


def _require_number_slots(
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


class ValidateAreaForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_area_form"

    async def required_slots(
        self,
        slots_mapped_in_domain: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Text]:
        shape = tracker.get_slot("shape")

        if not shape:
            return ["shape"]

        if shape == "rectangle":
            return ["shape", "length", "width"]
        if shape == "circle":
            return ["shape", "radius"]
        if shape == "triangle":
            return ["shape", "base", "height"]
        if shape == "trapezoid":
            return ["shape", "b1", "b2", "height"]
        if shape == "parallelogram":
            return ["shape", "base", "height"]

        return ["shape"]

    def validate_shape(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Дүрсээ сонгоно уу.")
            return {"shape": None}

        v = value.strip().lower()

        mapping = {
            "дөрвөлжин": "rectangle",
            "тэгш өнцөгт": "rectangle",
            "rectangle": "rectangle",
            "тойрог": "circle",
            "circle": "circle",
            "гурвалжин": "triangle",
            "triangle": "triangle",
            "трапец": "trapezoid",
            "trapezoid": "trapezoid",
            "параллелограмм": "parallelogram",
            "parallelogram": "parallelogram",
        }

        shape = mapping.get(v, v)
        if shape not in ["rectangle", "circle", "triangle", "trapezoid", "parallelogram"]:
            return {"shape": None}

        # шинэ дүрс сонгоход өмнөх хэмжээсүүдийг цэвэрлэх
        return {
            "shape": shape,
            "length": None, "width": None,
            "radius": None,
            "base": None, "height": None,
            "b1": None, "b2": None,
        }

    def _validate_pos(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if not _positive(num):
            dispatcher.utter_message(text=POSITIVE_NUMBER_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(num)}

    def validate_length(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("length", dispatcher, value)

    def validate_width(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("width", dispatcher, value)

    def validate_radius(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("radius", dispatcher, value)

    def validate_base(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("base", dispatcher, value)

    def validate_height(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("height", dispatcher, value)

    def validate_b1(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("b1", dispatcher, value)

    def validate_b2(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("b2", dispatcher, value)


class ValidateMathForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_math_form"

    async def required_slots(
        self,
        slots_mapped_in_domain: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Text]:
        task = tracker.get_slot("math_task")

        if not task:
            return ["math_task"]

        if task == "percent":
            return ["math_task", "percent_part", "percent_total"]
        if task == "ratio":
            return ["math_task", "ratio_a", "ratio_b"]
        if task == "proportion":
            return ["math_task", "prop_a", "prop_b", "prop_c"]
        if task == "speed":
            target = tracker.get_slot("speed_target")
            if not target:
                return ["math_task", "speed_target"]
            if target == "speed":
                return ["math_task", "speed_target", "distance_value", "time_value"]
            if target == "distance":
                return ["math_task", "speed_target", "speed_value", "time_value"]
            if target == "time":
                return ["math_task", "speed_target", "distance_value", "speed_value"]
            return ["math_task", "speed_target"]
        if task == "coordinate":
            return ["math_task", "coord_x", "coord_y"]
        if task == "composite_area":
            return ["math_task", "comp_len1", "comp_w1", "comp_len2", "comp_w2"]
        if task == "parallelogram":
            return ["math_task", "para_base", "para_height"]

        return ["math_task"]

    def validate_math_task(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Ямар бодлого бодохоо сонгоно уу.")
            return {"math_task": None}

        v = value.strip().lower()
        mapping = {
            "хувь": "percent",
            "percent": "percent",
            "харьцаа": "ratio",
            "ratio": "ratio",
            "пропорц": "proportion",
            "пропорцын бодлого": "proportion",
            "proportion": "proportion",
            "хурд": "speed",
            "зам-хурд-хугацаа": "speed",
            "хурд-зам-хугацаа": "speed",
            "speed": "speed",
            "координат": "coordinate",
            "координатын хавтгай": "coordinate",
            "coordinate": "coordinate",
            "нийлмэл талбай": "composite_area",
            "нийлмэл дүрс": "composite_area",
            "composite_area": "composite_area",
            "параллелограмм": "parallelogram",
            "parallelogram": "parallelogram",
        }

        task = mapping.get(v, v)
        allowed = {
            "percent",
            "ratio",
            "proportion",
            "speed",
            "coordinate",
            "composite_area",
            "parallelogram",
        }
        if task not in allowed:
            dispatcher.utter_message(text="Энэ төрлийн бодлого танигдсангүй. Дахин сонгоно уу.")
            return {"math_task": None}

        return {
            "math_task": task,
            "percent_part": None,
            "percent_total": None,
            "ratio_a": None,
            "ratio_b": None,
            "prop_a": None,
            "prop_b": None,
            "prop_c": None,
            "speed_target": None,
            "speed_value": None,
            "distance_value": None,
            "time_value": None,
            "coord_x": None,
            "coord_y": None,
            "comp_len1": None,
            "comp_w1": None,
            "comp_len2": None,
            "comp_w2": None,
            "para_base": None,
            "para_height": None,
        }

    def validate_speed_target(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Юуг олохоо сонгоно уу.")
            return {"speed_target": None}

        v = value.strip().lower()
        mapping = {
            "хурд": "speed",
            "v": "speed",
            "speed": "speed",
            "зам": "distance",
            "s": "distance",
            "distance": "distance",
            "хугацаа": "time",
            "t": "time",
            "time": "time",
        }

        target = mapping.get(v, v)
        if target not in {"speed", "distance", "time"}:
            dispatcher.utter_message(text="Хурд/зам/хугацаанаас сонгоно уу.")
            return {"speed_target": None}

        return {
            "speed_target": target,
            "speed_value": None,
            "distance_value": None,
            "time_value": None,
        }

    def _validate_pos(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if not _positive(num):
            dispatcher.utter_message(text=POSITIVE_NUMBER_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(num)}

    def _validate_any(self, slot_name: Text, dispatcher: CollectingDispatcher, value: Any) -> Dict[Text, Any]:
        num = _parse_number(str(value))
        if num is None:
            dispatcher.utter_message(text=NUMBER_MESSAGE)
            return {slot_name: None}
        return {slot_name: float(num)}

    def validate_percent_part(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("percent_part", dispatcher, value)

    def validate_percent_total(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("percent_total", dispatcher, value)

    def validate_ratio_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("ratio_a", dispatcher, value)

    def validate_ratio_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("ratio_b", dispatcher, value)

    def validate_prop_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("prop_a", dispatcher, value)

    def validate_prop_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("prop_b", dispatcher, value)

    def validate_prop_c(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("prop_c", dispatcher, value)

    def validate_speed_value(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("speed_value", dispatcher, value)

    def validate_distance_value(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("distance_value", dispatcher, value)

    def validate_time_value(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("time_value", dispatcher, value)

    def validate_coord_x(self, value: Any, dispatcher, tracker, domain):
        return self._validate_any("coord_x", dispatcher, value)

    def validate_coord_y(self, value: Any, dispatcher, tracker, domain):
        return self._validate_any("coord_y", dispatcher, value)

    def validate_comp_len1(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("comp_len1", dispatcher, value)

    def validate_comp_w1(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("comp_w1", dispatcher, value)

    def validate_comp_len2(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("comp_len2", dispatcher, value)

    def validate_comp_w2(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("comp_w2", dispatcher, value)

    def validate_para_base(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("para_base", dispatcher, value)

    def validate_para_height(self, value: Any, dispatcher, tracker, domain):
        return self._validate_pos("para_height", dispatcher, value)


class ActionCalculateArea(Action):
    def name(self) -> Text:
        return "action_calculate_area"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        shape = tracker.get_slot("shape")

        if shape == "rectangle":
            values = _require_positive_slots(dispatcher, tracker, ["length", "width"])
            if not values:
                return []
            length, width = values
            area = length * width
            dispatcher.utter_message(
                text=(
                    "Тэгш өнцөгт: "
                    f"S = урт x өргөн = {_fmt_num(length)} x {_fmt_num(width)} = {_fmt_num(area)}"
                )
            )

        elif shape == "circle":
            values = _require_positive_slots(dispatcher, tracker, ["radius"])
            if not values:
                return []
            r = values[0]
            area = math.pi * r * r
            dispatcher.utter_message(
                text=(
                    "Тойрог: "
                    f"S = pi * r^2 = pi * {_fmt_num(r)}^2 = {_fmt_num(area)}"
                )
            )

        elif shape == "triangle":
            values = _require_positive_slots(dispatcher, tracker, ["base", "height"])
            if not values:
                return []
            base, height = values
            area = 0.5 * base * height
            dispatcher.utter_message(
                text=(
                    "Гурвалжин: "
                    "S = 0.5 * суурь * өндөр = "
                    f"0.5 * {_fmt_num(base)} * {_fmt_num(height)} = {_fmt_num(area)}"
                )
            )

        elif shape == "trapezoid":
            values = _require_positive_slots(dispatcher, tracker, ["b1", "b2", "height"])
            if not values:
                return []
            b1, b2, h = values
            area = (b1 + b2) * h / 2.0
            dispatcher.utter_message(
                text=(
                    "Трапец: "
                    "S = (b1 + b2) * h / 2 = "
                    f"({_fmt_num(b1)} + {_fmt_num(b2)}) * {_fmt_num(h)} / 2 = {_fmt_num(area)}"
                )
            )

        elif shape == "parallelogram":
            values = _require_positive_slots(dispatcher, tracker, ["base", "height"])
            if not values:
                return []
            base, height = values
            area = base * height
            dispatcher.utter_message(
                text=(
                    "Параллелограмм: "
                    f"S = суурь * өндөр = {_fmt_num(base)} * {_fmt_num(height)} = {_fmt_num(area)}"
                )
            )

        else:
            dispatcher.utter_message(text="Дүрс сонгогдоогүй байна. 'дүрсний талбай ол' гэж дахин эхлүүлээрэй.")

        return []


class ActionCalculateMath(Action):
    def name(self) -> Text:
        return "action_calculate_math"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        task = tracker.get_slot("math_task")

        if task == "percent":
            values = _require_positive_slots(dispatcher, tracker, ["percent_part", "percent_total"])
            if not values:
                return []
            part, total = values
            percent = part / total * 100
            dispatcher.utter_message(
                text=(
                    "Хувь: хувь = (хэсэг / нийт) x 100% = "
                    f"({_fmt_num(part)} / {_fmt_num(total)}) x 100% = {_fmt_num(percent)}%"
                )
            )

        elif task == "ratio":
            values = _require_positive_slots(dispatcher, tracker, ["ratio_a", "ratio_b"])
            if not values:
                return []
            a, b = values
            ratio_value = a / b
            simplified = _simplify_ratio(a, b)
            simplified_text = f", хялбарчилбал {simplified}" if simplified else ""
            dispatcher.utter_message(
                text=(
                    "Харьцаа: "
                    f"{_fmt_num(a)}:{_fmt_num(b)}{simplified_text}, "
                    f"{_fmt_num(a)}/{_fmt_num(b)} = {_fmt_num(ratio_value)}"
                )
            )

        elif task == "proportion":
            values = _require_positive_slots(dispatcher, tracker, ["prop_a", "prop_b", "prop_c"])
            if not values:
                return []
            a, b, c = values
            x = b * c / a
            dispatcher.utter_message(
                text=(
                    "Пропорц: a:b = c:x => x = b*c/a = "
                    f"{_fmt_num(b)}*{_fmt_num(c)}/{_fmt_num(a)} = {_fmt_num(x)}"
                )
            )

        elif task == "speed":
            target = tracker.get_slot("speed_target")
            if target == "speed":
                values = _require_positive_slots(dispatcher, tracker, ["distance_value", "time_value"])
                if not values:
                    return []
                s, t = values
                v = s / t
                dispatcher.utter_message(
                    text=(
                        "Хурд: v = s / t = "
                        f"{_fmt_num(s)} / {_fmt_num(t)} = {_fmt_num(v)}"
                    )
                )
            elif target == "distance":
                values = _require_positive_slots(dispatcher, tracker, ["speed_value", "time_value"])
                if not values:
                    return []
                v, t = values
                s = v * t
                dispatcher.utter_message(
                    text=(
                        "Зам: s = v * t = "
                        f"{_fmt_num(v)} * {_fmt_num(t)} = {_fmt_num(s)}"
                    )
                )
            elif target == "time":
                values = _require_positive_slots(dispatcher, tracker, ["distance_value", "speed_value"])
                if not values:
                    return []
                s, v = values
                t = s / v
                dispatcher.utter_message(
                    text=(
                        "Хугацаа: t = s / v = "
                        f"{_fmt_num(s)} / {_fmt_num(v)} = {_fmt_num(t)}"
                    )
                )
            else:
                dispatcher.utter_message(text="Юуг олохоо сонгоно уу.")

        elif task == "coordinate":
            values = _require_number_slots(dispatcher, tracker, ["coord_x", "coord_y"])
            if not values:
                return []
            x, y = values
            if x == 0 and y == 0:
                position = "эх цэг"
            elif x == 0:
                position = "y тэнхлэг дээр"
            elif y == 0:
                position = "x тэнхлэг дээр"
            elif x > 0 and y > 0:
                position = "I квадрант"
            elif x < 0 and y > 0:
                position = "II квадрант"
            elif x < 0 and y < 0:
                position = "III квадрант"
            else:
                position = "IV квадрант"
            dispatcher.utter_message(
                text=f"Координат: A({_fmt_num(x)}, {_fmt_num(y)}) нь {position}."
            )

        elif task == "composite_area":
            values = _require_positive_slots(dispatcher, tracker, ["comp_len1", "comp_w1", "comp_len2", "comp_w2"])
            if not values:
                return []
            l1, w1, l2, w2 = values
            area = l1 * w1 + l2 * w2
            dispatcher.utter_message(
                text=(
                    "Нийлмэл талбай (2 тэгш өнцөгт): "
                    "S = l1*w1 + l2*w2 = "
                    f"{_fmt_num(l1)}*{_fmt_num(w1)} + {_fmt_num(l2)}*{_fmt_num(w2)} = {_fmt_num(area)}"
                )
            )

        elif task == "parallelogram":
            values = _require_positive_slots(dispatcher, tracker, ["para_base", "para_height"])
            if not values:
                return []
            b, h = values
            area = b * h
            dispatcher.utter_message(
                text=(
                    "Параллелограмм: S = b*h = "
                    f"{_fmt_num(b)}*{_fmt_num(h)} = {_fmt_num(area)}"
                )
            )

        else:
            dispatcher.utter_message(text="Бодлогын төрөл сонгогдоогүй байна.")

        return []


class ActionResetArea(Action):
    def name(self) -> Text:
        return "action_reset_area"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]


class ActionResetMath(Action):
    def name(self) -> Text:
        return "action_reset_math"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]
