"""
Menyiapkan master_dataset_labeled.parquet untuk dashboard.

Input (taruh satu folder dengan file ini):
  - master_dataset_indobert_final.csv   (dari Anggota 1)

Aturan sama persis dengan notebook Anggota 2:
  1. Hapus @mention dan link dari full_text_raw untuk mengukur teks
  2. Buang baris dengan minimal 2 huruf [ıİşŞğĞ] yang proporsinya minimal 2%
  3. Buang teks bersih kosong
  4. Pisahkan X menjadi X_Reply (teks diawali "@") dan X_Main
  5. Label sentimen = kolom indobert_sentiment
"""
import os
import pandas as pd


def report_rows(step, before, after):
    print(f"{step}: {before:,} -> {after:,} baris (dibuang {before - after:,})")


def is_turkish_noise(full_text_raw):
    text = (
        full_text_raw.astype(object)
        .fillna("")
        .str.replace(r"(?a)(@\w+|https?://\S+)", "", regex=True)
    )
    n_tr = text.str.count(r"[ıİşŞğĞ]")
    text_length = text.str.len().clip(lower=1)
    return n_tr.ge(2) & n_tr.div(text_length).ge(0.02)


df = pd.read_csv(
    "master_dataset_indobert_final.csv",
    dtype={"doc_id": str},
    keep_default_na=False,
)
print(f"CSV utama dimuat: {len(df):,} baris")
print(f"doc_id duplikat: {df['doc_id'].duplicated().sum():,}")
clean_text = df["full_text_clean"].astype("string").fillna("").str.strip()
print(f"full_text_clean literal 'nan'/'null': {clean_text.str.lower().isin(['nan', 'null']).sum():,}")
print(f"kata_kunci kosong: {df['kata_kunci'].astype('string').fillna('').str.strip().eq('').sum():,}")
print(f"created_at tidak valid: {pd.to_datetime(df['created_at'], errors='coerce').isna().sum():,}")

is_asing = is_turkish_noise(df["full_text_raw"])
print(f"Baris dibuang berdasarkan aturan teks: {int(is_asing.sum()):,}")
print("Dibuang per sumber:")
print(df.loc[is_asing, "sumber_data"].value_counts().to_string())
before = len(df)
df = df.loc[~is_asing].copy()
report_rows("Terapkan aturan teks asing", before, len(df))

before = len(df)
text = df["full_text_clean"].astype("string").fillna("").str.strip()
has_text = text.ne("")
df = df.loc[has_text].copy()
report_rows("Hapus full_text_clean kosong", before, len(df))

# Pisahkan X_Main / X_Reply
is_x = df["sumber_data"] == "X"
is_reply = is_x & df["full_text_raw"].astype(str).str.startswith("@")
before = len(df)
df.loc[is_reply, "sumber_data"] = "X_Reply"
df.loc[is_x & ~is_reply, "sumber_data"] = "X_Main"
report_rows("Klasifikasi sumber X (tidak menghapus baris)", before, len(df))

before = len(df)
df["sentiment"] = df["indobert_sentiment"]
report_rows("Salin indobert_sentiment (tidak menghapus baris)", before, len(df))

kolom = ["doc_id", "sumber_data", "kata_kunci", "created_at",
         "full_text_clean", "sentiment", "indobert_confidence"]
before = len(df)
df = df[[k for k in kolom if k in df.columns]].copy()
report_rows("Pilih kolom output (tidak menghapus baris)", before, len(df))
for k in ["sumber_data", "kata_kunci", "sentiment"]:
    df[k] = df[k].astype("category")

print(f"Total akhir: {len(df):,}")
print("Total per sumber:")
print(
    df["sumber_data"]
    .value_counts()
    .reindex(["X_Main", "X_Reply", "Threads"], fill_value=0)
    .to_string()
)
print("Total per sentimen:")
print(
    df["sentiment"]
    .value_counts()
    .reindex(["Negatif", "Netral", "Positif"], fill_value=0)
    .to_string()
)

df.to_parquet("master_dataset_labeled.parquet", index=False, compression="zstd")
print("Ukuran file: %.1f MB" % (os.path.getsize("master_dataset_labeled.parquet") / 1e6))