# Air Pollution Hybrid: GBM + Neural

Reproduction-with-swap of:

> Necula, C., Hauer, M., Fotache, D., Hurbean, E. (2025). *Advanced Hybrid Models
> for Air Pollution Forecasting: Combining SARIMA and BiLSTM Architectures.*
> Electronics 14(3), 549, MDPI. https://doi.org/10.3390/electronics14030549

Coursework: **Algorithms & Complexity**, S2 Informatika, Universitas Nusa Mandiri.

## Tujuan

Mengganti tahap-1 SARIMA pada hybrid rujukan dengan gradient boosting
(LightGBM/XGBoost), lalu membandingkan seluruh varian tahap-2 neural
(BiLSTM/Attention/Fourier) secara **apple-to-apple** dengan protokol identik.

## Hasil utama

- GBM+BiLSTM unggul di **9/9 sel** atas seluruh varian SARIMA.
- XGBoost+BiLSTM terbaik 6/9 sel (termasuk semua NO2).
- Peningkatan terbesar di horizon 1: PM25 0.329→0.141 (2.3×), PM10 0.410→0.187
  (2.2×), NO2 1.717→0.934 (1.8×).

## Repository map

| Path | Konten |
|---|---|
| `hybrid.py` | Pipeline: preprocessing (Box-Cox + MinMaxScaler), tahap-1 (SARIMA/LightGBM/XGBoost), tahap-2 (BiLSTM/Attention/Fourier) |
| `compare_ref_vs_gbm.py` | Perbandingan adil 45 kombinasi (5 tahap-1 × 3 polutan × 3 horizon) |
| `compare_fair.py` | Perbandingan tahap-2 (BiLSTM/Theta/TFT) |
| `run_full_bilstm.py` | Run penuh BiLSTM (18 kombinasi GBM × polutan × horizon) |
| `notebooks/Hybrid_GBM_Neural_executed.ipynb` | Notebook ala sajian rujukan, hasil tersemat |
| `results/` | Hasil full-run (RMSE/MAE/MAPE per kombinasi) + `figures/` (5 grafik) + `semua_hasil.json` |
| `Laporan_Eksperimen_Final.pdf` | Laporan lengkap semua eksperimen (4 hlm, 5 grafik), model final XGBoost+BiLSTM |

## Protokol (apple-to-apple)

- Test set 720 hari, horizon 1/7/30, window banding 120 hari.
- Box-Cox + MinMaxScaler hanya fit pada data latih (tanpa leakage).
- Metrik RMSE/MAE/MAPE.

## Cara menjalankan

```bash
python compare_ref_vs_gbm.py   # hasil → results/hasil_ref_vs_gbm.txt
python compare_fair.py         # hasil → results/hasil_fair_full.txt
# notebook: buka notebooks/Hybrid_GBM_Neural.ipynb
# (sel hasil membaca cache results/, cepat & reproduktibel)
```
