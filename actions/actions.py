import math
from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


def _parse_numbers(text: str) -> List[float]:
    if not text:
        return []
    cleaned = []
    allowed = set("0123456789.-")
    for ch in text:
        cleaned.append(ch if ch in allowed else " ")
    numbers: List[float] = []
    for part in "".join(cleaned).split():
        try:
            numbers.append(float(part))
        except ValueError:
            continue
    return numbers


class ActionAskSquareSide(Action):
    def name(self) -> str:
        return "action_ask_square_side"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_ask_square_side")
        return [SlotSet("pending_calc", "square_area")]


class ActionAskRectangleSides(Action):
    def name(self) -> str:
        return "action_ask_rectangle_sides"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_ask_rectangle_sides")
        return [SlotSet("pending_calc", "rectangle_area")]


class ActionAskCubeSide(Action):
    def name(self) -> str:
        return "action_ask_cube_side"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_ask_cube_side")
        return [SlotSet("pending_calc", "cube_volume")]


class ActionAskSphereRadius(Action):
    def name(self) -> str:
        return "action_ask_sphere_radius"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_ask_sphere_radius")
        return [SlotSet("pending_calc", "sphere_volume")]


class ActionAskConeParams(Action):
    def name(self) -> str:
        return "action_ask_cone_params"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_ask_cone_params")
        return [SlotSet("pending_calc", "cone_volume")]


class ActionAskCylinderParams(Action):
    def name(self) -> str:
        return "action_ask_cylinder_params"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        dispatcher.utter_message(response="utter_ask_cylinder_params")
        return [SlotSet("pending_calc", "cylinder_volume")]


class ActionHandleMeasurements(Action):
    def name(self) -> str:
        return "action_handle_measurements"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        pending = tracker.get_slot("pending_calc")
        numbers = _parse_numbers(tracker.latest_message.get("text", ""))

        if not pending:
            dispatcher.utter_message(response="utter_ask_calc_type")
            return []

        if pending == "square_area":
            if len(numbers) < 1:
                dispatcher.utter_message(response="utter_ask_square_side")
                return []
            side = numbers[0]
            if side <= 0:
                dispatcher.utter_message(text="Талын урт 0-ээс их байх хэрэгтэй.")
                return []
            area = side * side
            dispatcher.utter_message(
                text=f"Квадратын талбай: S = a^2 = {side}^2 = {area}"
            )
            return [SlotSet("pending_calc", None)]

        if pending == "rectangle_area":
            if len(numbers) < 2:
                dispatcher.utter_message(response="utter_ask_rectangle_sides")
                return []
            length, width = numbers[0], numbers[1]
            if length <= 0 or width <= 0:
                dispatcher.utter_message(text="Урт ба өргөн 0-ээс их байх хэрэгтэй.")
                return []
            area = length * width
            dispatcher.utter_message(
                text=(
                    f"Тэгш өнцөгтийн талбай: "
                    f"S = a*b = {length}*{width} = {area}"
                )
            )
            return [SlotSet("pending_calc", None)]

        if pending == "cube_volume":
            if len(numbers) < 1:
                dispatcher.utter_message(response="utter_ask_cube_side")
                return []
            side = numbers[0]
            if side <= 0:
                dispatcher.utter_message(text="Талын урт 0-ээс их байх хэрэгтэй.")
                return []
            volume = side ** 3
            dispatcher.utter_message(
                text=f"Кубын эзэлхүүн: V = a^3 = {side}^3 = {volume}"
            )
            return [SlotSet("pending_calc", None)]

        if pending == "sphere_volume":
            if len(numbers) < 1:
                dispatcher.utter_message(response="utter_ask_sphere_radius")
                return []
            radius = numbers[0]
            if radius <= 0:
                dispatcher.utter_message(text="Радиус 0-ээс их байх хэрэгтэй.")
                return []
            volume = (4.0 / 3.0) * math.pi * (radius ** 3)
            dispatcher.utter_message(
                text=(
                    f"Бөмбөрцгийн эзэлхүүн: "
                    f"V = 4/3*pi*r^3 = {volume}"
                )
            )
            return [SlotSet("pending_calc", None)]

        if pending == "cone_volume":
            if len(numbers) < 2:
                dispatcher.utter_message(response="utter_ask_cone_params")
                return []
            radius, height = numbers[0], numbers[1]
            if radius <= 0 or height <= 0:
                dispatcher.utter_message(text="Радиус ба өндөр 0-ээс их байх хэрэгтэй.")
                return []
            volume = (1.0 / 3.0) * math.pi * (radius ** 2) * height
            dispatcher.utter_message(
                text=(
                    f"Конусын эзэлхүүн: "
                    f"V = 1/3*pi*r^2*h = {volume}"
                )
            )
            return [SlotSet("pending_calc", None)]

        if pending == "cylinder_volume":
            if len(numbers) < 2:
                dispatcher.utter_message(response="utter_ask_cylinder_params")
                return []
            radius, height = numbers[0], numbers[1]
            if radius <= 0 or height <= 0:
                dispatcher.utter_message(text="Радиус ба өндөр 0-ээс их байх хэрэгтэй.")
                return []
            volume = math.pi * (radius ** 2) * height
            dispatcher.utter_message(
                text=(
                    f"Цилиндрийн эзэлхүүн: "
                    f"V = pi*r^2*h = {volume}"
                )
            )
            return [SlotSet("pending_calc", None)]

        dispatcher.utter_message(response="utter_ask_calc_type")
        return []
