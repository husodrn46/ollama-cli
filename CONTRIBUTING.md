# Katki Rehberi

Ollama CLI Pro'ya katki yaptiginiz icin tesekkurler! Bu rehber, projeye nasil katki yapabileceginizi aciklar.

## Baslamadan Once

1. Projeyi fork'layin
2. Yerel kopyanizi klonlayin:
   ```bash
   git clone https://github.com/KULLANICI_ADINIZ/ollama-cli.git
   cd ollama-cli
   ```
3. Gelistirme ortamini kurun:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

## Gelistirme Sureci

### 1. Yeni Branch Olusturun

```bash
git checkout -b feature/ozellik-adi
# veya
git checkout -b fix/hata-aciklamasi
```

### 2. Degisikliklerinizi Yapin

- Kod stilini koruyun (mevcut kodla tutarli olun)
- Type hint'leri kullanin
- Onemli fonksiyonlara docstring ekleyin
- Gereksiz bagimlilik eklemeyin

### 3. Testleri Calistirin

```bash
pytest
```

Yeni ozellikler icin test ekleyin.

### 4. Commit Mesaji

Anlasilir commit mesajlari yazin:

```
feat: Yeni ozellik aciklamasi
fix: Duzeltilen hata aciklamasi
refactor: Yeniden yapilandirma aciklamasi
docs: Dokumantasyon guncellemesi
test: Test ekleme/guncelleme
```

### 5. Pull Request

1. Degisikliklerinizi push'layin:
   ```bash
   git push origin feature/ozellik-adi
   ```
2. GitHub'da Pull Request olusturun
3. PR aciklamasinda degisiklikleri detayli anlatın

## Kod Standartlari

### Python Stili

- Python 3.10+ ozellikleri kullanilabilir
- Type hint'ler tercih edilir
- Fonksiyonlar tek bir is yapmali
- Maksimum satir uzunlugu: 100 karakter

### Dosya Yapisi

```
ollama_cli/
├── app.py           # Ana uygulama
├── commands.py      # Komut handler'lari
├── chat_engine.py   # Sohbet mantigi
├── model_manager.py # Model islemleri
├── ui_display.py    # Terminal UI
├── models.py        # Pydantic modelleri
├── storage.py       # Dosya islemleri
├── session_store.py # Oturum yonetimi
├── security.py      # Sifreleme
└── ...
```

### Test Yazma

```python
# tests/test_modul.py
import pytest
from ollama_cli.modul import fonksiyon

def test_fonksiyon_basarili():
    sonuc = fonksiyon(girdi)
    assert sonuc == beklenen

def test_fonksiyon_hata():
    with pytest.raises(ValueError):
        fonksiyon(gecersiz_girdi)
```

## Hata Raporlama

GitHub Issues uzerinden hata raporlayin:

1. Hatanin acik bir tanimi
2. Yeniden olusturma adimlari
3. Beklenen ve gerceklesen davranis
4. Python versiyonu ve isletim sistemi
5. Hata mesaji (varsa)

## Ozellik Onerileri

GitHub Issues uzerinden onerin:

1. Ozelligin amaci
2. Kullanim senaryosu
3. Olasi implementasyon fikirleri (opsiyonel)

## Lisans

Katkilariniz MIT lisansi altinda yayinlanacaktir.

## Sorular?

GitHub Issues veya Discussions uzerinden sorabilirsiniz.

Katkiniz icin tesekkurler!
