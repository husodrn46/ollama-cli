# Changelog

Bu dosya projedeki onemli degisiklikleri belgeler.
Format [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) standardina,
versiyonlama [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standardina uygundur.

## [5.1.0] - 2024-01-16

### Eklenenler
- Prompt kutuphanesi sistemi (`/prompt` komutlari)
- Otomatik baslik olusturma (sohbet icerigi bazli)
- Clipboard izleme ozelligi (`clipboard_monitor`)
- Canli TPS (tokens per second) gostergesi
- Gelismis markdown rendering

### Degisiklikler
- Performans iyilestirmeleri
- Daha iyi hata mesajlari

## [5.0.0] - 2024-01-14

### Eklenenler
- Moduler mimari (chat_engine, model_manager, ui_display ayri moduller)
- Kapsamli test altyapisi (77 test fonksiyonu)
- Pydantic ile yapilandirma dogrulamasi
- Sifreleme destegi (Fernet)
- Session store ile oturum yonetimi
- Benchmark sistemi (`/bench`)
- Model karsilastirma (`/compare`)
- Profil yonetimi (`/profile`)

### Degisiklikler
- Komut handler'lari `commands.py`'ye tasindi
- `app.py` sadece ana uygulama mantigi iceriyor
- Tum config/data dosyalari XDG standart dizinlerine tasindi

### Duzeltmeler
- Context token yonetimi iyilestirildi
- Model capability tespiti duzeltildi

## [4.0.0] - 2024-01-10

### Eklenenler
- Vision modelleri destegi (`/img`, `/paste`)
- Favoriler ve sablonlar (`/fav`, `/tpl`)
- Tema sistemi
- Coklu satir girisi (Alt+Enter)
- Token takibi

### Degisiklikler
- Rich ve prompt_toolkit entegrasyonu
- Yeni komut sistemi

## [3.0.0] - 2024-01-05

### Eklenenler
- Model yonetimi (`/pull`, `/delete`)
- Sohbet kaydetme/yukleme (`/save`, `/load`)
- Export (`/export html|md|json|txt`)

## [2.0.0] - 2024-01-01

### Eklenenler
- Streaming yanit destegi
- Sistem prompt'u destegi
- `/retry`, `/edit`, `/copy` komutlari

## [1.0.0] - 2023-12-20

### Eklenenler
- Ilk surum
- Temel sohbet fonksiyonelitesi
- Model secimi
- Basit komut sistemi
