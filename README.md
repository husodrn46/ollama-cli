# Ollama CLI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/husodrn46/ollama-cli/actions/workflows/test.yml/badge.svg)](https://github.com/husodrn46/ollama-cli/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/husodrn46/ollama-cli/graph/badge.svg)](https://codecov.io/gh/husodrn46/ollama-cli)

Ollama modelleri ile terminal uzerinden etkilesim kurmak icin gelistirilmis, zengin ozelliklere sahip bir komut satiri arayuzu (CLI).

## Proje Açıklaması ve Amacı

Ollama CLI, yerel olarak çalışan büyük dil modelleriyle sohbet etmeyi kolaylaştıran, kullanıcı dostu ve güçlü bir araçtır. `rich` ve `prompt_toolkit` gibi kütüphaneler sayesinde modern ve interaktif bir terminal deneyimi sunar. Amacı, geliştiricilerin ve meraklıların Ollama'nın gücünü verimli bir şekilde komut satırından kullanabilmelerini sağlamaktır.

## Demo

```
╔══════════════════════════════════════════════════════════════╗
║  ◉ OLLAMA CLI PRO v5.1                                       ║
║  ⚡ http://localhost:11434  🎨 dark                          ║
╚══════════════════════════════════════════════════════════════╝

Model: llama3.2 (3.2B) [vision] [context: 128K]
✓ Sistem promptu yuklendi
👁️ Vision model - /img ile resim gonder
↑↓ gecmis • Tab tamamla • /help yardim • /q cikis
─────────────────────────────────────────────────────────────────

◉ SEN: Python ile basit bir web scraper yaz

◉ LLAMA3.2
─────────────────────────────────────────────────────────────────
İşte basit bir web scraper örneği:

```python
import requests
from bs4 import BeautifulSoup

def scrape(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.title.string

print(scrape('https://example.com'))
```

Bu kod:
1. `requests` ile sayfayı indirir
2. `BeautifulSoup` ile parse eder
3. Sayfa başlığını döndürür

⏱ 2.3s  ◈ 156 token  ⚡ 67.8 t/s
```

### Özellik Gösterimi

| Özellik | Komut | Açıklama |
|---------|-------|----------|
| 🎨 Temalar | `/theme ocean` | 4 farklı renk teması |
| 📋 Favoriler | `/fav add kod "Kod yaz:"` | Hızlı prompt kısayolları |
| 🖼️ Görsel | `/img foto.png` | Vision modellere resim gönder |
| 💾 Kaydet | `/save sohbet.json` | Sohbeti dışa aktar |
| 📊 Benchmark | `/bench all` | Model performans testi |
| 🔒 Güvenlik | `/security mask on` | Hassas veri maskeleme |

## Temel Özellikler

-   **İnteraktif Sohbet:** Akıcı ve gerçek zamanlı (streaming) bir sohbet arayüzü.
-   **Model Yönetimi:** Sistemde yüklü Ollama modellerini listeleme, seçme, indirme (`/pull`) ve silme (`/delete`).
-   **Gelişmiş Komutlar:** `/help`, `/model`, `/save`, `/load`, `/retry`, `/edit`, `/copy` gibi 30'dan fazla komut.
-   **Görsel Desteği (Vision):** Vision modelleri ile yerel dosyalardan (`/img`) veya panodan (`/paste`) resim göndererek sohbet etme.
-   **Özelleştirme:** Temalar, sistem prompt'ları ve personelar.
-   **Favoriler ve Şablonlar:** Sık kullanılan prompt'ları (`/fav`) ve değişkenli şablonları (`/tpl`) kaydetme ve kullanma.
-   **Sohbet Yönetimi:** Sohbet geçmişini kaydetme ve `html`, `md`, `json`, `txt` formatlarında dışa aktarma.
-   **Oturum Yönetimi:** Kaydedilen sohbetleri listeleme, etiketleme, silme; otomatik kayıt ve saklama politikaları.
-   **Context Yönetimi:** Token bütçesi, otomatik özetleme ve manuel `/summarize`.
-   **Profil Yönetimi:** Modele özel sıcaklık/prompt profilleri ve `/profile` ile hızlı geçiş.
-   **Güvenlik:** Maskeleme ve opsiyonel şifreli kayıt/export.
-   **Markdown Görünümü:** Yanıtları Markdown olarak render eder; `**kalın**` vb. düzgün görünür.
-   **Model Yetenekleri:** Context limiti, vision/tools/embedding rozetleri ve uyumsuzluk uyarıları.
-   **Benchmark:** `/bench` ile model performansını ölç ve sonuçları kaydet.
-   **Diğer Özellikler:** Çoklu satır girişi, token takibi, model karşılaştırma (`/compare`), panoya kopyalama.

## Teknoloji Stack'i

-   **Dil:** Python
-   **Temel Kütüphaneler:** `requests`, `rich`, `prompt_toolkit`, `pydantic`, `platformdirs`, `cryptography`

## Kurulum Adımları

1.  **Repository'yi klonlayın:**
    ```bash
    git clone <repository-url>
    cd ollama-cli
    ```

2.  **Bağımlılıkları yükleyin:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    Alternatif olarak:
    ```bash
    pip install -e .
    ```

3.  **Aracı başlatın:**
    ```bash
    ./ollama-chat
    ```
    İsteğe bağlı olarak, bu betiği PATH'inize ekleyerek (`sudo cp ollama-chat /usr/local/bin/`) sistem genelinde `ollama-chat` komutuyla çalıştırabilirsiniz.

## Yapılandırma

-   **Config yolu:** `~/.config/ollama-cli-pro/config.json`
-   **Favoriler/şablonlar:** `~/.local/share/ollama-cli-pro/favorites.json`
-   **Geçmiş:** `~/.local/share/ollama-cli-pro/history.txt`
-   **Log:** `~/.local/share/ollama-cli-pro/ollama-cli.log`
-   **Oturumlar:** `~/.local/share/ollama-cli-pro/sessions/`
-   **Model cache:** `~/.local/share/ollama-cli-pro/model_cache.json`
-   **Benchmark sonuçları:** `~/.local/share/ollama-cli-pro/benchmarks.json`

İlk çalıştırmada mevcut `config.json`, `prompts.json`, `favorites.json` dosyaları yeni konuma taşınır.

### Ortam Değişkenleri

-   `OLLAMA_HOST`: Ollama API adresini geçersiz kılar.
-   `OLLAMA_CLI_HOME`: Tüm config/data dosyaları için kök dizin.
-   `OLLAMA_CLI_KEY`: Şifreli kayıt/export için anahtar.

### Oturum ve Saklama

-   `auto_save`: Yanıt sonrası otomatik kayıt.
-   `session_retention_count`: Maksimum oturum sayısı.
-   `session_retention_days`: Maksimum saklama süresi (gün).

### Context Yönetimi

-   `context_token_budget`: Tahmini token bütçesi.
-   `context_keep_last`: Özetten sonra korunacak mesaj sayısı.
-   `context_autosummarize`: Otomatik özetleme.
-   `summary_model`: Özetleme için model (opsiyonel).

### Profil Yönetimi

```json
{
  "profiles": {
    "fast": {
      "model": "ministral",
      "temperature": 0.2,
      "system_prompt": "Kisa ve net yanit ver.",
      "auto_apply": false
    }
  },
  "model_profiles": {
    "gpt-oss": {
      "temperature": 0.7,
      "system_prompt": "Daha detayli acikla."
    }
  }
}
```

### Güvenlik

-   `mask_sensitive`: Kayıt/export öncesi hassas verileri maskele.
-   `encryption_enabled`: Oturum kayıtlarını şifrele.
-   `encrypt_exports`: Export dosyalarını şifrele.

Şifreleme için `/security keygen` veya `OLLAMA_CLI_KEY` kullanabilirsiniz.

### Markdown

-   `render_markdown`: Yanıtları Markdown olarak render et (varsayılan: `true`).

İsterseniz `/markdown on|off` ile anında açıp kapatabilirsiniz.

### Benchmark

-   `benchmark_prompt`: Benchmark promptu.
-   `benchmark_runs`: Tek model için tekrar sayısı.
-   `benchmark_timeout`: Timeout (saniye).
-   `benchmark_temperature`: Benchmark sıcaklığı.

### Diagnostik Mod

-   Başlangıçta: `./ollama-chat --diag`
-   Çalışırken: `/diag on` veya `/diag off`

## Kullanım

Aracı başlattıktan sonra mevcut modeller listelenir ve bir model seçmeniz istenir. Ardından doğrudan sohbete başlayabilirsiniz.

-   Yardım menüsünü görmek için `/help` yazın.
-   Modeli değiştirmek için `/model` yazın.
-   Oturumları listelemek için `/sessions` yazın.
-   Profil seçmek için `/profile <isim>` yazın.
-   Sohbeti sonlandırmak için `/quit` yazın.

Detaylı komut listesi için `docs/cheatsheet.md` dosyasına bakabilirsiniz.

## Dosya/Klasör Yapısı

```
.
├── ollama_cli/         # Paket kodu
├── ollama_chat.py      # Legacy entrypoint
├── ollama-chat         # Bash launcher
├── config.json         # Legacy config (ilk calistirmada tasinir)
├── prompts.json        # Legacy promptlar (ilk calistirmada tasinir)
├── favorites.json      # Legacy favoriler (ilk calistirmada tasinir)
├── pyproject.toml
├── requirements.txt
└── docs/
    └── cheatsheet.md
```
