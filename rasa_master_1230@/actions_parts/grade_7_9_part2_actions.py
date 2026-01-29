from typing import Any, Dict, List, Text, Optional
import math
import re

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import AllSlotsReset

NUMBER_MESSAGE = "Please enter a number. (e.g., 2, -3, 4.5)"
POSITIVE_MESSAGE = "Please enter a positive number. (e.g., 2, 4.5)"


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


def _fmt_num(value: float) -> str:
    return f"{value:g}"


class ValidateMath8Form(FormValidationAction):
    def name(self) -> Text:
        return "validate_math8_form"

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

        if topic == "identity":
            return ["topic", "identity_type", "id_a", "id_b"]

        if topic == "quadratic":
            return ["topic", "quad_a", "quad_b", "quad_c"]

        if topic == "system":
            return ["topic", "sys_a1", "sys_b1", "sys_c1", "sys_a2", "sys_b2", "sys_c2"]

        if topic == "circle":
            return ["topic", "circle_r"]

        if topic == "trig":
            return ["topic", "trig_opp", "trig_adj", "trig_hyp"]

        return ["topic"]

    def validate_topic(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Please choose a topic.")
            return {"topic": None}

        v = value.strip().lower()
        mapping = {
            "identity": "identity",
            "quadratic": "quadratic",
            "system": "system",
            "circle": "circle",
            "trig": "trig",
            "square identity": "identity",
            "quadratic equation": "quadratic",
            "linear system": "system",
            "circle formula": "circle",
            "trigonometry": "trig",
            "??????? ??????????": "identity",
            "??????? ?????????": "quadratic",
            "??????": "system",
            "??????": "circle",
            "???????????": "trig",
        }

        topic = mapping.get(v, v)
        allowed = {"identity", "quadratic", "system", "circle", "trig"}
        if topic not in allowed:
            dispatcher.utter_message(text="Unknown topic. Choose again.")
            return {"topic": None}

        return {
            "topic": topic,
            "identity_type": None,
            "id_a": None,
            "id_b": None,
            "quad_a": None,
            "quad_b": None,
            "quad_c": None,
            "sys_a1": None,
            "sys_b1": None,
            "sys_c1": None,
            "sys_a2": None,
            "sys_b2": None,
            "sys_c2": None,
            "circle_r": None,
            "trig_opp": None,
            "trig_adj": None,
            "trig_hyp": None,
        }

    def validate_identity_type(self, value: Text, dispatcher, tracker, domain) -> Dict[Text, Any]:
        if not value:
            dispatcher.utter_message(text="Choose an identity type.")
            return {"identity_type": None}

        v = value.strip().lower().replace(" ", "")
        mapping = {
            "(a+b)^2": "square_sum",
            "(a-b)^2": "square_diff",
            "a^2-b^2": "diff_squares",
            "square_sum": "square_sum",
            "square_diff": "square_diff",
            "diff_squares": "diff_squares",
            "squareofsum": "square_sum",
            "squareofdiff": "square_diff",
            "differenceofsquares": "diff_squares",
        }

        identity_type = mapping.get(v, v)
        if identity_type not in {"square_sum", "square_diff", "diff_squares"}:
            dispatcher.utter_message(text="Unknown identity type.")
            return {"identity_type": None}

        return {"identity_type": identity_type}

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

    def validate_id_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("id_a", dispatcher, value)

    def validate_id_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("id_b", dispatcher, value)

    def validate_quad_a(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("quad_a", dispatcher, value)

    def validate_quad_b(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("quad_b", dispatcher, value)

    def validate_quad_c(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("quad_c", dispatcher, value)

    def validate_sys_a1(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("sys_a1", dispatcher, value)

    def validate_sys_b1(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("sys_b1", dispatcher, value)

    def validate_sys_c1(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("sys_c1", dispatcher, value)

    def validate_sys_a2(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("sys_a2", dispatcher, value)

    def validate_sys_b2(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("sys_b2", dispatcher, value)

    def validate_sys_c2(self, value: Any, dispatcher, tracker, domain):
        return self._validate_number("sys_c2", dispatcher, value)

    def validate_circle_r(self, value: Any, dispatcher, tracker, domain):
        return self._validate_positive("circle_r", dispatcher, value)

    def validate_trig_opp(self, value: Any, dispatcher, tracker, domain):
        return self._validate_positive("trig_opp", dispatcher, value)

    def validate_trig_adj(self, value: Any, dispatcher, tracker, domain):
        return self._validate_positive("trig_adj", dispatcher, value)

    def validate_trig_hyp(self, value: Any, dispatcher, tracker, domain):
        return self._validate_positive("trig_hyp", dispatcher, value)


class ActionCalculateMath8(Action):
    def name(self) -> Text:
        return "action_calculate_math8"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        topic = tracker.get_slot("topic")

        if topic == "identity":
            identity_type = tracker.get_slot("identity_type")
            a = tracker.get_slot("id_a")
            b = tracker.get_slot("id_b")
            if a is None or b is None:
                dispatcher.utter_message(text=NUMBER_MESSAGE)
                return []
            a = float(a)
            b = float(b)

            if identity_type == "square_sum":
                value = (a + b) ** 2
                dispatcher.utter_message(
                    text=(
                        "(a + b)^2 = a^2 + 2ab + b^2. "
                        f"For a={_fmt_num(a)}, b={_fmt_num(b)}: (a+b)^2 = {_fmt_num(value)}"
                    )
                )
                return []

            if identity_type == "square_diff":
                value = (a - b) ** 2
                dispatcher.utter_message(
                    text=(
                        "(a - b)^2 = a^2 - 2ab + b^2. "
                        f"For a={_fmt_num(a)}, b={_fmt_num(b)}: (a-b)^2 = {_fmt_num(value)}"
                    )
                )
                return []

            if identity_type == "diff_squares":
                value = a ** 2 - b ** 2
                dispatcher.utter_message(
                    text=(
                        "a^2 - b^2 = (a-b)(a+b). "
                        f"For a={_fmt_num(a)}, b={_fmt_num(b)}: a^2-b^2 = {_fmt_num(value)}"
                    )
                )
                return []

            dispatcher.utter_message(text="Unknown identity type.")
            return []

        if topic == "quadratic":
            a = tracker.get_slot("quad_a")
            b = tracker.get_slot("quad_b")
            c = tracker.get_slot("quad_c")
            if a is None or b is None or c is None:
                dispatcher.utter_message(text=NUMBER_MESSAGE)
                return []
            a = float(a)
            b = float(b)
            c = float(c)

            if abs(a) < 1e-9:
                if abs(b) < 1e-9:
                    dispatcher.utter_message(text="Not a valid equation (a=0 and b=0).")
                    return []
                x = -c / b
                dispatcher.utter_message(text=f"Linear equation: x = {_fmt_num(x)}")
                return []

            d = b * b - 4 * a * c
            if d > 0:
                sqrt_d = math.sqrt(d)
                x1 = (-b - sqrt_d) / (2 * a)
                x2 = (-b + sqrt_d) / (2 * a)
                dispatcher.utter_message(
                    text=(
                        f"D = {_fmt_num(d)}. Two real roots: x1 = {_fmt_num(x1)}, x2 = {_fmt_num(x2)}"
                    )
                )
                return []
            if abs(d) < 1e-9:
                x = -b / (2 * a)
                dispatcher.utter_message(text=f"D = 0. One real root: x = {_fmt_num(x)}")
                return []

            dispatcher.utter_message(text=f"D = {_fmt_num(d)} < 0. No real roots.")
            return []

        if topic == "system":
            a1 = tracker.get_slot("sys_a1")
            b1 = tracker.get_slot("sys_b1")
            c1 = tracker.get_slot("sys_c1")
            a2 = tracker.get_slot("sys_a2")
            b2 = tracker.get_slot("sys_b2")
            c2 = tracker.get_slot("sys_c2")
            if None in (a1, b1, c1, a2, b2, c2):
                dispatcher.utter_message(text=NUMBER_MESSAGE)
                return []
            a1 = float(a1)
            b1 = float(b1)
            c1 = float(c1)
            a2 = float(a2)
            b2 = float(b2)
            c2 = float(c2)

            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-9:
                dispatcher.utter_message(text="No unique solution (determinant is 0).")
                return []

            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            dispatcher.utter_message(
                text=f"Solution: x = {_fmt_num(x)}, y = {_fmt_num(y)}"
            )
            return []

        if topic == "circle":
            r = tracker.get_slot("circle_r")
            if r is None:
                dispatcher.utter_message(text=POSITIVE_MESSAGE)
                return []
            r = float(r)
            c = 2 * math.pi * r
            s = math.pi * r * r
            dispatcher.utter_message(
                text=(
                    f"C = 2*pi*r = {_fmt_num(c)}; S = pi*r^2 = {_fmt_num(s)}"
                )
            )
            return []

        if topic == "trig":
            opp = tracker.get_slot("trig_opp")
            adj = tracker.get_slot("trig_adj")
            hyp = tracker.get_slot("trig_hyp")
            if None in (opp, adj, hyp):
                dispatcher.utter_message(text=POSITIVE_MESSAGE)
                return []
            opp = float(opp)
            adj = float(adj)
            hyp = float(hyp)

            if abs(hyp) < 1e-9 or abs(adj) < 1e-9:
                dispatcher.utter_message(text="Invalid sides for trig ratios.")
                return []

            sin_v = opp / hyp
            cos_v = adj / hyp
            tan_v = None if abs(adj) < 1e-9 else opp / adj

            if tan_v is None:
                dispatcher.utter_message(
                    text=(
                        f"sin = {_fmt_num(sin_v)}, cos = {_fmt_num(cos_v)}, tan is undefined"
                    )
                )
                return []

            dispatcher.utter_message(
                text=(
                    f"sin = {_fmt_num(sin_v)}, cos = {_fmt_num(cos_v)}, tan = {_fmt_num(tan_v)}"
                )
            )
            return []

        dispatcher.utter_message(text="Unknown topic.")
        return []


class ActionResetMath8(Action):
    def name(self) -> Text:
        return "action_reset_math8"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]
