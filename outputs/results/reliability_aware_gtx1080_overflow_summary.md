# Reliability-Aware Overflow Training Summary (GTX1080 x2)

Ditulis: 18 Agustus 2026, setelah kedua GTX1080 menyelesaikan 30 run overflow reliability_aware
(15 run masing-masing, seed 0-4 x ablation stage A1/A2/A3/A4/A5/Full).

## Catatan metodologi
- Semua 30 run ini dilatih di GTX1080 (GPU0/GPU1), BUKAN di A4000, menggunakan
  `configs/config_gpu1080_reliability.yaml` (batch_size=16, bukan 32 seperti default config.yaml).
- Sesuai catatan di config tsb: kalau A4000 nanti juga mencapai (seed, stage) yang sama, train.py
  akan skip (checkpoint dari 1080 dianggap final) — jadi angka final untuk run-run ini datang dari
  batch_size=16, bukan 32. Perlu disebut di limitations paper.
- Semua nilai adalah **val macro F1 di kondisi modality lengkap (C,D,M)** — BUKAN hasil evaluasi
  under missing/contradictory modality (itu tahap Phase 6-7 terpisah, pakai evaluate.py, belum
  dijalankan).

## Tabel lengkap (30 run)

| Seed | A1 | A2 | A3 | A4 | A5 | Full |
|---|---|---|---|---|---|---|
| 0 | 0.5279 | 0.5016 | 0.5396 | 0.5109 | 0.5957 | 0.5549 |
| 1 | 0.4823 | 0.4702 | 0.5002 | 0.4732 | 0.4917 | 0.4768 |
| 2 | 0.5075 | 0.4979 | 0.4923 | 0.4911 | 0.4427 | 0.5093 |
| 3 | 0.4623 | 0.4968 | 0.4749 | 0.4343 | 0.4813 | 0.4332 |
| 4 | 0.5050 | 0.5485 | 0.5226 | 0.5308 | 0.5288 | 0.5254 |

## Rata-rata per ablation stage (n=5 seed)

| Stage | Mean | SD | Min | Max |
|---|---|---|---|---|
| A1 (baseline) | 0.4970 | 0.0252 | 0.4623 | 0.5279 |
| A2 (+reliability weighting) | 0.5030 | 0.0283 | 0.4702 | 0.5485 |
| A3 (+modality dropout) | 0.5059 | 0.0255 | 0.4749 | 0.5396 |
| A4 (+disagreement) | 0.4881 | 0.0370 | 0.4343 | 0.5308 |
| A5 (+calibration) | 0.5080 | 0.0578 | 0.4427 | 0.5957 |
| Full (+abstention) | 0.4999 | 0.0468 | 0.4332 | 0.5549 |

Overall (30 run): mean = 0.5003, SD = 0.0358.

## Perbandingan ke baseline (Phase 3 unimodal, Phase 4 naive_concat CD)

| Baseline | Mean F1 |
|---|---|
| Unimodal D (dermoscopy sendiri) | 0.502 |
| Unimodal C (clinical sendiri) | 0.410 |
| Unimodal M (metadata sendiri) | 0.183 |
| naive_concat C,D | ~0.489 |
| **reliability_aware (semua stage, overall)** | **0.500** |

## Observasi

1. Tren A1 -> A3 naik konsisten (0.497 -> 0.506), sesuai arah hipotesis (makin lengkap komponen
   reliability, makin baik). Tapi A4 justru turun (0.488), dan A5/Full variansnya jauh lebih besar
   (SD 0.058 dan 0.047 vs A1 yang cuma 0.025) — bukan tren monoton bersih begitu semua seed
   lengkap.
2. Di kondisi modality lengkap (C,D,M), reliability_aware secara keseluruhan (mean 0.500) kurang
   lebih SETARA dengan unimodal D sendirian (0.502) dan sedikit di atas naive_concat CD (~0.489) —
   belum ada keunggulan besar dari raw F1 di kondisi ini. Ini konsisten dengan arahan CLAUDE.md
   ("Do not chase SOTA accuracy") — klaim utama proyek ada di ketahanan saat modality
   hilang/kontradiktif (Phase 6-7 evaluate.py), bukan di angka F1 kondisi lengkap ini.
3. Outlier yang perlu dilirik saat error analysis: seed2/A5 (F1=0.4427, early-stop di epoch 9 —
   jauh lebih awal dari run lain yang biasanya 15-40 epoch) dan seed3/Full (F1=0.4332, early-stop
   epoch 13). Kemungkinan seed/init kurang beruntung, bukan tanda bug sistematis (run lain di
   ablation stage yang sama untuk seed lain normal).
4. A5 (yang menambah komponen kalibrasi) punya varians antar-seed paling tinggi dari semua stage
   (SD 0.058) — worth dicek apakah komponen kalibrasi ini sensitif terhadap random init, bisa jadi
   temuan tersendiri di paper, bukan cuma noise.

## File sumber

- `outputs/logs/gpu0_1080_rel_run.log`
- `outputs/logs/gpu1_1080_rel_run.log`
- `outputs/checkpoints/reliability_aware_CDM_seed{0-4}_{A1,A2,A3,A4,A5,Full}_best.pt` (30 file)
- Data tabel mentah: `reliability_aware_gtx1080_overflow_summary.csv` (file sebelah)
