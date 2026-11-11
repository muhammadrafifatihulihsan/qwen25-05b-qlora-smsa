# Keputusan lisensi dan atribusi

Status: keputusan sementara untuk keperluan tugas akademik, wajib diperiksa ulang
sebelum dataset didistribusikan ulang ke publik.

## Dataset SmSA (IndoNLU)

Sumber: https://github.com/IndoNLP/indonlu dan https://huggingface.co/datasets/indonlp/indonlu

Keterangan yang ditemukan tidak seragam: berkas LICENSE repositori menyebut Apache 2.0,
badge README menyebut MIT, dan loader SmSA pada NusaCrowd menyebut CC BY-SA 4.0.

Keputusan saya:

- Data dipakai untuk analisis dan pelaporan akademik, bukan produk komersial.
- Berkas data mentah tidak disertakan di repositori kode. Yang disertakan hanya data
  olahan hasil audit, sehingga pemakaian ulang tetap merujuk ke sumber asli.
- Untuk distribusi ulang dataset di masa depan, saya akan memakai ketentuan paling ketat
  yaitu CC BY-SA 4.0 beserta atribusi, atau meminta konfirmasi pemelihara IndoNLU.
  Lisensi seluruh repositori tidak otomatis berlaku sebagai lisensi dataset.

## Model Qwen2.5-0.5B-Instruct

Sumber: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Keterangan: Apache 2.0 pada kartu model.

Keputusan saya:

- Bobot model tidak disertakan di repositori. Skrip `src/03_download_model.py` mengunduh
  dari sumber resmi sehingga pemakai menerima ketentuan aslinya.
- Bobot adaptor hasil pelatihan juga tidak disertakan agar repositori tetap ringan dan
  bebas berkas biner besar. Adaptor dapat dibangun ulang dari konfigurasi yang tersedia.
- Jika adaptor didistribusikan kelak, saya akan mencantumkan bahwa adaptor diturunkan
  dari model berlisensi Apache 2.0 beserta tautan sumbernya.

## Pustaka perangkat lunak

PyTorch, Transformers, PEFT, bitsandbytes, Accelerate, pandas, NumPy, scikit-learn,
Matplotlib, dan huggingface-hub memakai lisensi permisif (MIT, BSD, atau Apache 2.0).
Versi yang benar-benar terpasang dicatat pada `requirements.txt`.

Lisensi kode penelitian ini adalah MIT. Lisensi tersebut tidak mencakup dataset SmSA.

## Batas tanggung jawab

Penulis tidak menjamin kelayakan hukum pemakaian ulang dataset oleh pihak lain. Setiap
pemakai ulang bertanggung jawab memeriksa sendiri ketentuan yang berlaku saat pemakaian.
