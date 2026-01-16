# Ollama CLI Pro Cheatsheet

## Temel Komutlar

- `/help` - Yardım menüsü
- `/quit` veya `/q` - Çıkış
- `/clear` veya `/c` - Sohbeti temizle
- `/model` veya `/m` - Model seç
- `/info` veya `/i` - Model bilgisi

## Favoriler ve Şablonlar

- `/fav` - Favorileri listele
- `/fav add <isim> <prompt>` - Favori ekle
- `/fav <isim> [metin]` - Favoriyi çalıştır
- `/tpl` - Şablonları listele
- `/tpl <isim> [degisken=deger]` - Şablon çalıştır

## Sohbet Yönetimi

- `/save` veya `/s` - Sohbeti kaydet
- `/load` veya `/l` - Kayıtlı sohbet seç
- `/sessions` - Kayıtlı sohbetleri listele
- `/session list|open|tag|untag|rename|delete` - Oturum yönetimi
- `/history` veya `/h` - Mesaj geçmişi
- `/title <baslik>` - Sohbete başlık ver
- `/export <html|json|txt|md>` - Dışa aktar

## Model İşlemleri

- `/pull <model>` - Model indir
- `/delete <model>` - Model sil
- `/stats` - Yüklü modeller ve VRAM
- `/compare` - Modelleri karşılaştır
- `/bench [all] [prompt]` - Basit benchmark çalıştır
- `/quick <model>` - Sohbeti koruyarak model değiştir

## Gelişmiş

- `/retry` - Son yanıtı yeniden üret
- `/edit` - Son kullanıcı mesajını düzenle
- `/copy` - Son yanıtı panoya kopyala
- `/search <kelime>` - Mesajlarda ara
- `/tokens` - Token kullanımını göster
- `/context` - Context durumunu göster
- `/summarize` - Konuşmayı özetle
- `/persona <isim|off>` - Persona değiştir
- `/profile <isim|off>` - Profil seç
- `/security ...` - Güvenlik ayarları
- `/temp <0.0-2.0|off>` - Sıcaklık ayarı
- `/diag [on|off]` - Diagnostik mod
- `/markdown on|off` - Markdown görünümünü aç/kapat

## Görsel

- `/img <yol> [soru]` - Dosyadan resim gönder
- `/paste [soru]` - Panodan resim gönder

## Örnekler

```bash
/fav add ozet "Bu metni ozetle:"
/fav ozet Uzun bir metin...

/tpl kod-aciklama dil=python kod="print('Merhaba')"

/export html
/compare
```
