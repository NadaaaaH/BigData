"""
Menyiapkan master_dataset_labeled.parquet untuk dashboard.

Input (taruh satu folder dengan file ini):
  - master_dataset_indobert_final.csv   (dari Anggota 1)
  - 0_baris_dibuang_asing.csv           (dari Anggota 2, folder hasil_analisis_anggota2_pyspark)

Aturan sama persis dengan notebook Anggota 2:
  1. Buang 3.512 baris tweet asing (Turki)
  2. Buang teks bersih kosong
  3. Pisahkan X menjadi X_Reply (teks diawali "@") dan X_Main
  4. Label sentimen = kolom indobert_sentiment
Hasil yang diharapkan: 158.185 baris (X_Main 79.613, X_Reply 43.978, Threads 34.594).
"""
import os
import pandas as pd

df = pd.read_csv("master_dataset_indobert_final.csv", dtype={"doc_id": str})
print("Baris awal:", len(df))

if os.path.exists("0_baris_dibuang_asing.csv"):
    asing = pd.read_csv("0_baris_dibuang_asing.csv", dtype={"doc_id": str})
    df = df[~df["doc_id"].isin(asing["doc_id"])]
else:
    print("PERINGATAN: 0_baris_dibuang_asing.csv tidak ditemukan, baris asing tidak dibuang.")

df = df[df["full_text_clean"].notna() & (df["full_text_clean"].astype(str).str.strip() != "")]

# Pisahkan X_Main / X_Reply
is_x = df["sumber_data"] == "X"
is_reply = is_x & df["full_text_raw"].astype(str).str.startswith("@")
df.loc[is_reply, "sumber_data"] = "X_Reply"
df.loc[is_x & ~is_reply, "sumber_data"] = "X_Main"

df["sentiment"] = df["indobert_sentiment"]

kolom = ["doc_id", "sumber_data", "kata_kunci", "created_at",
         "full_text_clean", "sentiment", "indobert_confidence"]
df = df[[k for k in kolom if k in df.columns]].copy()
for k in ["sumber_data", "kata_kunci", "sentiment"]:
    df[k] = df[k].astype("category")

print("Baris akhir:", len(df))
print(df["sumber_data"].value_counts())
print(df["sentiment"].value_counts())

df.to_parquet("master_dataset_labeled.parquet", index=False, compression="zstd")
print("Ukuran file: %.1f MB" % (os.path.getsize("master_dataset_labeled.parquet") / 1e6))