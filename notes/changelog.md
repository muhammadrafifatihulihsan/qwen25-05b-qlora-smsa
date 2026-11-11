# Changelog penelitian

Catat setiap perubahan rancangan atau penyesuaian skrip: tanggal, apa yang berubah, alasan, dan apakah berlaku untuk semua kondisi.

| Tanggal | Perubahan | Alasan | Berlaku untuk |
|---|---|---|---|
| 11 November 2026 | Lingkungan eksekusi dipindahkan dari Python 3.10 ke Python 3.13 | `import peft` gagal (segmentation fault) pada Python 3.10.10 dengan torch 2.14.1 | Semua kondisi |
| 11 November 2026 | torch 2.14.1+cu126, transformers 5.22.2, peft 0.21.2, bitsandbytes 0.50.2 | Versi GPU untuk RTX 2050 4 GB; peft dan bitsandbytes berjalan normal di 3.13 | Semua kondisi |
| 11 November 2026 | Unduhan model memakai `HF_ENDPOINT=https://hf-mirror.com` | Unduhan langsung dari huggingface.co berhenti pada 27 MB karena jaringan lambat | Tahap persiapan saja |
| 11 November 2026 | max_len tetap 256, tidak diturunkan ke 128 | Pilot tidak OOM: puncak alokasi 1.402,8 MiB dan puncak reserved 2.824,0 MiB; rasio data terpotong 0,0 | Semua kondisi |
| 11 November 2026 | Pelatihan penuh dijalankan untuk seed 42 pada empat rank | Data uji belum dibuka; seed tambahan menunggu waktu tersedia | seed 42 |
| 11 November 2026 | Seluruh 12 run protokol selesai (rank 4, 8, 16, 32 x seed 42, 123, 2026) | Menuntaskan rancangan satu faktor sebelum data uji dibuka | Semua kondisi |
| 11 November 2026 | Data uji dievaluasi sekali setelah rank dibekukan pada rank 32 | Aturan pemilihan: rank terkecil dengan selisih Macro-F1 validasi tidak lebih dari 0,01; rank 16 berjarak 0,0140 sehingga di luar toleransi | Tahap uji akhir |
| 11 November 2026 | Uji tambahan rank 64 dengan tiga seed, di luar protokol | Melihat apakah kenaikan masih berlanjut di atas rank 32. Hasilnya tidak dipakai untuk memilih konfigurasi karena aturan sudah dibekukan | Tambahan saja |
| 11 November 2026 | README, CITATION, dan .gitignore dirapikan; laporan tidak lagi disimpan di repo | Repo berisi kode, data, dan hasil analisis; laporan akademik dikumpulkan terpisah melalui SMILE | Dokumentasi |

## Catatan pilot (11 November 2026)

- Perintah: `py -3.13 train.py --rank 32 --seed 42 --out ../runs/pilot_r32_s42 --pilot`
- Status: `ok`
- Parameter terlatih pada rank 32: 2.162.688
- Memori setelah model dimuat: 704,99 MiB
- Puncak alokasi: 1.402,77 MiB; puncak reserved: 2.824,00 MiB
- Waktu 20 langkah: 66,84 detik, yaitu 3,34 detik per langkah
- Loss langkah terakhir: 0,2244; rata-rata 10 langkah terakhir: 0,2918
- Rasio data latih terpotong pada 256 token: 0,0
- Perkiraan waktu satu pelatihan penuh (375 langkah): sekitar 21 menit
