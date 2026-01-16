from ollama_cli.security import decrypt_text, encrypt_text, generate_key, mask_sensitive_text


def test_mask_sensitive_text():
    text = "api_key=ABCDEF1234567890"
    masked = mask_sensitive_text(text, [r"api_key=[A-Za-z0-9]+"])
    assert "REDACTED" in masked


def test_encrypt_roundtrip():
    key = generate_key()
    cipher = encrypt_text("hello", key)
    plain = decrypt_text(cipher, key)
    assert plain == "hello"
