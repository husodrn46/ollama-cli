from ollama_cli.utils import estimate_message_tokens, get_model_prompt


def test_get_model_prompt_default():
    prompts = {"_default": {"system_prompt": "base"}}
    prompt = get_model_prompt("unknown:latest", prompts)
    assert prompt["system_prompt"] == "base"


def test_estimate_message_tokens():
    msg = {"content": "hello world"}
    assert estimate_message_tokens(msg) > 0
