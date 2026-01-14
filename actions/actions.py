from typing import Any, Dict, List, Text, Optional
import math
import re

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import AllSlotsReset

POSITIVE_NUMBER_MESSAGE = "Эерэг тоо оруулна уу. (ж: 5 эсвэл 3.5)"


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


class ActionResetArea(Action):
    def name(self) -> Text:
        return "action_reset_area"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]
