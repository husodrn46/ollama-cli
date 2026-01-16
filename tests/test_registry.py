from ollama_cli.commands import Command, CommandRegistry


def test_registry_aliases():
    registry = CommandRegistry()
    registry.register(Command("/help", ("/h",), "Help", None, lambda _: True))

    assert registry.get("/help") is not None
    assert registry.get("/h") is not None
