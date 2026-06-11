# Himdex

Open-source efficient multimodal starter model for text and image experiments.

Himdex adalah starter model machine learning multimodal: satu backbone Transformer
ringan yang bisa menerima dataset teks dan gambar.

Target besarnya: model pribadi yang efisien, fleksibel, dan bisa terus ditingkatkan.
Target realistis MVP ini: bisa dilatih untuk klasifikasi teks dari CSV dan klasifikasi
gambar dari folder dataset.

## Kenapa Mulai Dari Sini?

Tidak ada model yang otomatis paling bagus untuk semua dataset. Dataset gambar,
teks pendek, teks panjang, data medis, data finansial, dan data noisy semuanya punya
kebutuhan berbeda. Jadi Himdex dibuat dengan filosofi:

- satu encoder inti yang hemat;
- adapter input berbeda untuk teks dan gambar;
- format training sederhana;
- mudah diganti jadi versi lebih besar;
- bisa ditambah contrastive learning, pretraining, LoRA, distillation, dan retrieval.

## Arsitektur MVP

```text
Teks UTF-8 -> ByteTokenizer -> token embedding \
                                                 -> shared Transformer -> classifier
Gambar     -> image patches  -> patch embedding /
```

Keuntungan byte tokenizer:

- tidak perlu training tokenizer dulu;
- bisa membaca banyak bahasa;
- cocok untuk eksperimen awal.

Keuntungan patch embedding:

- gambar diperlakukan sebagai urutan token;
- bisa memakai encoder yang sama dengan teks.

## Instalasi

```powershell
git clone https://github.com/dexpie/himdex.git
cd himdex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Training Teks

Buat CSV dengan minimal kolom `text` dan `label`.

Contoh:

```csv
text,label
"aku suka produk ini",positif
"ini buruk sekali",negatif
```

Jalankan:

```powershell
himdex-train --task text-classification --data data\reviews.csv --epochs 5 --batch-size 16
```

Untuk mencoba cepat dengan dataset mini bawaan:

```powershell
himdex-train --task text-classification --data examples\text_toy.csv --epochs 1 --batch-size 2 --hidden-size 64 --num-layers 1 --max-text-length 32
```

Kalau nama kolom beda:

```powershell
himdex-train --task text-classification --data data\reviews.csv --text-column kalimat --label-column kelas
```

## Training Gambar

Gunakan struktur `ImageFolder`:

```text
data/images/
  kucing/
    a.jpg
    b.jpg
  anjing/
    c.jpg
    d.jpg
```

Jalankan:

```powershell
himdex-train --task image-classification --data data\images --epochs 5 --batch-size 16
```

Checkpoint akan tersimpan sebagai:

```text
runs/himdex/himdex.pt
```

## Dataset Starter 5GB

Untuk mulai memakai campuran dataset publik dari GitHub, Hugging Face, torchvision,
dan Kaggle opsional, lihat:

[DATASETS.md](DATASETS.md)

Perintah ringkas:

```powershell
himdex-prepare-data --root data\himdex_starter --budget-gb 5
himdex-pretrain-text --data data\himdex_starter\prepared --epochs 1
```

## Checkpoint Starter

Repo ini menyertakan checkpoint demo kecil:

```text
checkpoints/himdex_text_starter.pt
```

Checkpoint ini dilatih dengan masked byte prediction dari dataset starter lokal.
Ini berguna untuk demo dan eksperimen lanjutan, bukan klaim model final.

Lanjutkan training dari checkpoint:

```powershell
himdex-pretrain-text --data data\himdex_starter\prepared --resume-from checkpoints\himdex_text_starter.pt --epochs 1 --batch-size 8 --max-steps 200
```

## Membuat Himdex Lebih "Peak"

Urutan pengembangan yang paling masuk akal:

1. **MVP supervised**
   Latih klasifikasi teks dan gambar seperti starter ini.

2. **Pretraining teks**
   Tambahkan masked byte prediction atau causal language modeling supaya Himdex
   punya pemahaman bahasa dasar.

3. **Pretraining gambar**
   Tambahkan masked image modeling atau self-supervised contrastive learning.

4. **Alignment teks-gambar**
   Tambahkan contrastive loss seperti CLIP agar teks dan gambar hidup di embedding
   space yang sama.

5. **Efisiensi**
   Tambahkan LoRA, quantization, pruning, gradient checkpointing, dan knowledge
   distillation.

6. **Dataset router**
   Buat sistem yang otomatis mendeteksi jenis dataset, memilih head, dan menyiapkan
   preprocessing.

7. **Evaluation suite**
   Ukur akurasi, latency, ukuran model, VRAM, robustness, dan generalisasi.

## Prinsip Penting

Kalau tujuanmu adalah membuat model yang benar-benar kuat, jangan mulai dari model
raksasa. Mulai dari model kecil yang bisa:

- dilatih sampai selesai;
- diuji dengan jelas;
- dibandingkan dengan baseline;
- diperbaiki berkali-kali.

Himdex versi awal ini adalah fondasi itu.
