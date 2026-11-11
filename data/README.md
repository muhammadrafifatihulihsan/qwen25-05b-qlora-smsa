# Data

Folder ini hanya memuat data olahan, bukan data mentah penuh.

- `processed/train.csv` subset latih 2000 baris hasil stratifikasi (seed sampel 2026),
  sudah melewati audit duplikat dan pemisahan dari validasi serta uji.
- `processed/valid.csv` 1260 baris, dipakai memilih rank.
- `processed/test.csv` 500 baris, hanya dibuka setelah rank dibekukan.
- `processed/laporan_duplikat.json` catatan audit: jumlah duplikat dihapus dan hash berkas.
- `processed/indeks_subset_2000.json` indeks baris data latih asli yang dipakai.

Sumber: SmSA (IndoNLU). Berkas asli tidak disertakan agar repositori tetap ringan;
unduh ulang lewat `src/01_download_data.py` lalu jalankan `src/02_prepare_data.py`
untuk membangun ulang berkas di folder ini. Hash pada `laporan_duplikat.json`
memungkinkan pemeriksaan bahwa hasilnya identik.

Lisensi dataset berbeda dari lisensi kode. Baca `notes/lisensi.md` sebelum
mendistribusikan ulang data.
