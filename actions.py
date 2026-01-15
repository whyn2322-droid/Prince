from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


def get_numbers(tracker):
    a = tracker.get_slot("a")
    b = tracker.get_slot("b")
    return a, b


class ActionAdd(Action):
    def name(self) -> Text:
        return "action_add"

    def run(self, dispatcher, tracker, domain):
        a, b = get_numbers(tracker)
        if a is None or b is None:
            dispatcher.utter_message("Хоёр тоо өгнө үү.")
            return []
        dispatcher.utter_message(f"Хариу = {float(a) + float(b)}")
        return []


class ActionSubtract(Action):
    def name(self) -> Text:
        return "action_subtract"

    def run(self, dispatcher, tracker, domain):
        a, b = get_numbers(tracker)
        if a is None or b is None:
            dispatcher.utter_message("Хоёр тоо өгнө үү.")
            return []
        dispatcher.utter_message(f"Хариу = {float(a) - float(b)}")
        return []


class ActionMultiply(Action):
    def name(self) -> Text:
        return "action_multiply"

    def run(self, dispatcher, tracker, domain):
        a, b = get_numbers(tracker)
        if a is None or b is None:
            dispatcher.utter_message("Хоёр тоо өгнө үү.")
            return []
        dispatcher.utter_message(f"Хариу = {float(a) * float(b)}")
        return []


class ActionDivide(Action):
    def name(self) -> Text:
        return "action_divide"

    def run(self, dispatcher, tracker, domain):
        a, b = get_numbers(tracker)
        if a is None or b is None:
            dispatcher.utter_message("Хоёр тоо өгнө үү.")
            return []
        if float(b) == 0:
            dispatcher.utter_message("0-д хуваах боломжгүй.")
            return []
        dispatcher.utter_message(f"Хариу = {float(a) / float(b)}")
        return []


class ActionRemainder(Action):
    def name(self) -> Text:
        return "action_remainder"

    def run(self, dispatcher, tracker, domain):
        a, b = get_numbers(tracker)
        if a is None or b is None:
            dispatcher.utter_message("Хоёр тоо өгнө үү.")
            return []
        dispatcher.utter_message(f"Үлдэгдэл = {float(a) % float(b)}")
        return []
