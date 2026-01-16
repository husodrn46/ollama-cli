import os

import pytest
import requests


def test_ollama_tags():
    host = os.environ.get("OLLAMA_HOST")
    if not host:
        pytest.skip("OLLAMA_HOST ayarlanmadigi icin entegrasyon testi atlandi")

    response = requests.get(f"{host}/api/tags", timeout=10)
    assert response.status_code == 200
