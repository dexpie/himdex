# Himdex Dataset Starter 5GB

Tujuan fase ini adalah memberi Himdex campuran data awal yang cukup beragam tanpa
menghabiskan storage. Batas awal: **5GB**.

## Realita Penting

"Semua dataset dari Kaggle, GitHub, Hugging Face, atau internet" tidak mungkin
dipakai secara literal di mesin lokal kecil. Jumlahnya sangat besar, lisensinya
berbeda-beda, dan banyak dataset Kaggle membutuhkan login. Strategi Himdex:

- mulai dari registry dataset publik yang aman untuk eksperimen awal;
- streaming dataset Hugging Face agar tidak langsung mengunduh semuanya;
- batasi jumlah baris/gambar per dataset;
- tulis manifest agar sumber data bisa diaudit;
- gunakan pretraining dulu untuk data campuran, baru fine-tuning untuk tugas khusus.

## Dataset Awal

Registry default saat ini:

- `github_tiny_shakespeare`: teks kecil untuk language-model smoke test.
- `hf_ag_news`: teks berita untuk klasifikasi topik.
- `hf_imdb`: teks review film untuk sentiment.
- `hf_wikitext_2`: teks Wikipedia untuk language modeling.
- `hf_beans`: gambar tanaman untuk klasifikasi gambar.
- `hf_cifar10`: gambar natural 10 kelas via Hugging Face streaming.
- `torchvision_mnist`: gambar digit.
- `torchvision_fashion_mnist`: gambar fashion.
- `torchvision_cifar10`: gambar natural 10 kelas, tersedia di registry tapi tidak
  masuk default karena download torchvision bisa gagal di beberapa environment.
- `kaggle_sms_spam`: teks spam SMS, opsional karena butuh token Kaggle.

## Instalasi

```powershell
cd C:\Users\Gading\Documents\Codex\2026-06-03\aku-pengen-buat-model-buat-machine\outputs\himdex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Cek Rencana Tanpa Download

```powershell
himdex-prepare-data --budget-gb 5 --dry-run
```

## Download Dataset Starter

```powershell
himdex-prepare-data --root data\himdex_starter --budget-gb 5
```

Output penting:

```text
data/himdex_starter/manifest.json
data/himdex_starter/prepared/text_classification/*.csv
data/himdex_starter/prepared/text_lm/*.txt
data/himdex_starter/prepared/image_classification/
data/himdex_starter/raw/torchvision/
```

Kalau mau lebih agresif tapi tetap di bawah 5GB:

```powershell
himdex-prepare-data --root data\himdex_starter --budget-gb 5 --rows-per-text-dataset 200000 --images-per-image-dataset 50000
```

## Kaggle

Kaggle biasanya butuh token akun.

Opsi environment variable:

```powershell
$env:KAGGLE_USERNAME="username_kamu"
$env:KAGGLE_KEY="api_key_kamu"
himdex-prepare-data --include-sources github,hf,torchvision,kaggle --budget-gb 5
```

Opsi file:

```text
C:\Users\<nama-kamu>\.kaggle\kaggle.json
```

Setelah token ada, dataset Kaggle di registry akan diproses otomatis jika sumber
`kaggle` dimasukkan lewat `--include-sources`.

## Pretraining Teks Campuran

Jalankan dari dataset hasil prepare:

```powershell
himdex-pretrain-text --data data\himdex_starter\prepared --epochs 1 --batch-size 16 --max-text-length 256
```

Pretraining ini memakai masked byte prediction. Ini cocok untuk mencampur data teks
dari banyak sumber karena tidak bergantung pada label dataset.

## Fine-Tuning

Setelah pretraining, gunakan dataset tugas spesifik.

Contoh teks:

```powershell
himdex-train --task text-classification --data data\himdex_starter\prepared\text_classification\hf_ag_news.csv --epochs 3
```

Contoh gambar:

```powershell
himdex-train --task image-classification --data data\images --epochs 3
```

Catatan: jangan langsung mencampur semua dataset klasifikasi menjadi satu training
classification biasa, karena arti label antar dataset tidak sama.
