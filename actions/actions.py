from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Text

from rasa_sdk import Action, FormValidationAction, Tracker
from rasa_sdk.events import AllSlotsReset, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

NUMBER_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)?")

SHAPE_SYNONYMS = {
    "circle": {"circle", "тойрог", "тойргийн", "дугуй"},
    "rectangle": {"rectangle", "тэгш өнцөгт", "тэгш өнцөгтийн"},
    "square": {"square", "дөрвөлжин", "квадрат"},
    "triangle": {"triangle", "гурвалжин", "гурвалжны"},
}

SHAPE_LABELS = {
    "circle": "тойрог",
    "rectangle": "тэгш өнцөгт",
    "square": "дөрвөлжин",
    "triangle": "гурвалжин",
}


def normalize_shape(value: Optional[Text]) -> Optional[Text]:
    if not value:
        return None
    text = value.strip().lower()
    for shape, synonyms in SHAPE_SYNONYMS.items():
        if text == shape or text in synonyms:
            return shape
    for shape, synonyms in SHAPE_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in text:
                return shape
    return None


def parse_positive_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = NUMBER_PATTERN.search(str(value))
        if not match:
            return None
        number = float(match.group(0).replace(",", "."))
    if number <= 0:
        return None
    return number


class ValidatePerimeterForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_perimeter_form"

    async def required_slots(
        self,
        slots_mapped_in_domain: List[Text],
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Text]:
        shape = normalize_shape(tracker.get_slot("shape"))
        if shape == "circle":
            return ["shape", "radius"]
        if shape == "rectangle":
            return ["shape", "width", "height"]
        if shape == "square":
            return ["shape", "square_side"]
        if shape == "triangle":
            return ["shape", "side_a", "side_b", "side_c"]
        return ["shape"]

    def validate_shape(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        shape = normalize_shape(value)
        if shape:
            return {"shape": shape}
        dispatcher.utter_message(response="utter_invalid_shape")
        return {"shape": None}

    def validate_radius(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_positive_number(value)
        if number is not None:
            return {"radius": number}
        dispatcher.utter_message(response="utter_invalid_number")
        return {"radius": None}

    def validate_width(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_positive_number(value)
        if number is not None:
            return {"width": number}
        dispatcher.utter_message(response="utter_invalid_number")
        return {"width": None}

    def validate_height(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_positive_number(value)
        if number is not None:
            return {"height": number}
        dispatcher.utter_message(response="utter_invalid_number")
        return {"height": None}

    def validate_square_side(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_positive_number(value)
        if number is not None:
            return {"square_side": number}
        dispatcher.utter_message(response="utter_invalid_number")
        return {"square_side": None}

    def validate_side_a(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_positive_number(value)
        if number is not None:
            return {"side_a": number}
        dispatcher.utter_message(response="utter_invalid_number")
        return {"side_a": None}

    def validate_side_b(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_positive_number(value)
        if number is not None:
            return {"side_b": number}
        dispatcher.utter_message(response="utter_invalid_number")
        return {"side_b": None}

    def validate_side_c(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        number = parse_positive_number(value)
        if number is not None:
            return {"side_c": number}
        dispatcher.utter_message(response="utter_invalid_number")
        return {"side_c": None}


class ActionCalculatePerimeter(Action):
    def name(self) -> Text:
        return "action_calculate_perimeter"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:
        shape = normalize_shape(tracker.get_slot("shape"))
        if not shape:
            dispatcher.utter_message(response="utter_invalid_shape")
            return []

        perimeter = None
        details = None
        if shape == "circle":
            radius = tracker.get_slot("radius")
            if radius is not None:
                perimeter = 2 * math.pi * radius
                details = (
                    "Томьёо: P = 2 * pi * радиус\n"
                    f"Тооцоолол: P = 2 * {math.pi:.4f} * {radius:.2f} = {perimeter:.2f}"
                )
        elif shape == "rectangle":
            width = tracker.get_slot("width")
            height = tracker.get_slot("height")
            if width is not None and height is not None:
                perimeter = 2 * (width + height)
                details = (
                    "Томьёо: P = 2(урт + өргөн)\n"
                    f"Тооцоолол: P = 2({width:.2f} + {height:.2f}) = {perimeter:.2f}"
                )
        elif shape == "square":
            side = tracker.get_slot("square_side")
            if side is not None:
                perimeter = 4 * side
                details = (
                    "Томьёо: P = 4 * тал\n"
                    f"Тооцоолол: P = 4 * {side:.2f} = {perimeter:.2f}"
                )
        elif shape == "triangle":
            side_a = tracker.get_slot("side_a")
            side_b = tracker.get_slot("side_b")
            side_c = tracker.get_slot("side_c")
            if side_a is not None and side_b is not None and side_c is not None:
                perimeter = side_a + side_b + side_c
                details = (
                    "Томьёо: P = a + b + c\n"
                    f"Тооцоолол: P = {side_a:.2f} + {side_b:.2f} + {side_c:.2f} = {perimeter:.2f}"
                )

        if perimeter is None:
            dispatcher.utter_message(text="Дутуу утга байна. Дахин оролдоно уу.")
            return []

        label = SHAPE_LABELS.get(shape, "дүрс")
        dispatcher.utter_message(text=f"{label} дүрсний хүрээ {perimeter:.2f} байна.")
        if details:
            dispatcher.utter_message(text=details)
        return [AllSlotsReset(), FollowupAction("perimeter_form")]
