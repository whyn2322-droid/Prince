import asyncio
import pathlib

from rasa.core.agent import Agent
from rasa.core.utils import AvailableEndpoints


def latest_model_path(models_dir: pathlib.Path) -> str:
    candidates = sorted(models_dir.glob("*.tar.gz"))
    if not candidates:
        raise SystemExit("models/ дотор загвар олдсонгүй. Эхлээд сургана уу.")
    return str(candidates[-1])


async def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    model_path = latest_model_path(project_root / "models")
    endpoints_path = project_root / "endpoints.yml"
    action_endpoint = None
    if endpoints_path.exists():
        endpoints = AvailableEndpoints.read_endpoints(str(endpoints_path))
        action_endpoint = endpoints.action_endpoint

    agent = Agent.load(model_path, action_endpoint=action_endpoint)

    print("Монгол хэлээр бичээд чатлаарай. Гарах бол 'exit' эсвэл 'гарах' гэж бичнэ үү.")
    while True:
        text = input("> ").strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit", "гарах", "боллоо"}:
            print("Баяртай!")
            break
        responses = await agent.handle_text(text)
        for response in responses:
            message = response.get("text")
            if message:
                print(message)


if __name__ == "__main__":
    asyncio.run(main())
