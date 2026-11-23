import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Radar Opini Publik: Kinerja Polri",
    page_icon=":material/analytics:",
    layout="wide",
)

DATA_FILE = "master_dataset_labeled.parquet"
HASIL_DIR = "hasil"  # file pendukung dari Anggota 1 & 2
KEYWORDS = ["polri", "polisi", "polda", "acab", "isilop", "parcok"]
SRC_ORDER = ["X_Main", "X_Reply", "Threads"]
ORDER = ["Negatif", "Netral", "Positif"]
COLOR_MAP = {"Negatif": "#FF6B6B", "Netral": "#FFD166", "Positif": "#06D6A0"}


def hasil(*names):
    """Path file pertama yang ada di folder hasil/, atau None."""
    for n in names:
        p = os.path.join(HASIL_DIR, n)
        if os.path.exists(p):
            return p
    return None


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_FILE)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    for c in ["sumber_data", "kata_kunci", "sentiment"]:
        df[c] = df[c].astype(str)
    df["sentiment"] = df["sentiment"].str.strip().str.capitalize()
    df["bulan"] = df["created_at"].dt.to_period("M").dt.to_timestamp()
    low = df["kata_kunci"].str.lower()
    for kw in KEYWORDS:
        df[f"kw_{kw}"] = low.str.contains(kw, regex=False)
    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat {DATA_FILE}: {e}")
    st.stop()

tgl_min, tgl_max = df["created_at"].min().date(), df["created_at"].max().date()

# ---------- HEADER ----------
st.title("Radar Opini Publik: Kinerja Polri")
st.caption(
    f"Analisis sentimen publik di X dan Threads era Presiden Prabowo Subianto "
    f"({tgl_min:%d %b %Y} – {tgl_max:%d %b %Y})"
)

# ---------- SIDEBAR ----------
st.sidebar.header("Filter Eksplorasi")
rentang = st.sidebar.date_input(
    "Rentang tanggal:", value=(tgl_min, tgl_max), min_value=tgl_min, max_value=tgl_max
)
tgl_a, tgl_b = rentang if isinstance(rentang, tuple) and len(rentang) == 2 else (tgl_min, tgl_max)

src_all = [s for s in SRC_ORDER if s in df["sumber_data"].unique()]
src_opt = st.sidebar.multiselect("Sumber data:", src_all, default=src_all)
kw_all = [k for k in KEYWORDS if df[f"kw_{k}"].any()]
kw_opt = st.sidebar.multiselect("Kata kunci:", kw_all, default=kw_all)
sent_all = [s for s in ORDER if s in df["sentiment"].unique()]
sent_opt = st.sidebar.multiselect("Sentimen:", sent_all, default=sent_all)

st.sidebar.caption(
    "Unggahan yang ditemukan lewat beberapa kata kunci dihitung di tiap kata kunci tersebut."
)

mask_kw = pd.Series(False, index=df.index)
for k in kw_opt:
    mask_kw |= df[f"kw_{k}"]

df_f = df[
    (df["created_at"].dt.date >= tgl_a)
    & (df["created_at"].dt.date <= tgl_b)
    & (df["sumber_data"].isin(src_opt))
    & df["sentiment"].isin(sent_opt)
    & mask_kw
]

if df_f.empty:
    st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

# ---------- KPI ----------
total = len(df_f)
cnt = df_f["sentiment"].value_counts()


def pct(label):
    return f"{cnt.get(label, 0) / total * 100:.1f}%"


c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Data", f"{total:,}")
c2.metric("Negatif", f"{cnt.get('Negatif', 0):,}", pct("Negatif"), delta_color="inverse")
c3.metric("Netral", f"{cnt.get('Netral', 0):,}", pct("Netral"), delta_color="off")
c4.metric("Positif", f"{cnt.get('Positif', 0):,}", pct("Positif"))
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Distribusi", "Tren Waktu", "Sumber & Kata Kunci", "Model, Statistik & Topik", "Data"]
)

# ---------- TAB 1 ----------
with tab1:
    dist = df_f["sentiment"].value_counts().reindex(ORDER).dropna().reset_index()
    dist.columns = ["sentiment", "jumlah"]
    left, right = st.columns(2)
    with left:
        st.subheader("Proporsi Sentimen")
        st.plotly_chart(
            px.pie(dist, names="sentiment", values="jumlah", hole=0.45,
                   color="sentiment", color_discrete_map=COLOR_MAP),
            width="stretch",
        )
    with right:
        st.subheader("Jumlah per Kelas Sentimen")
        fig = px.bar(dist, x="sentiment", y="jumlah", color="sentiment", text="jumlah",
                     color_discrete_map=COLOR_MAP, labels={"sentiment": "Sentimen", "jumlah": "Jumlah"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ---------- TAB 2 ----------
with tab2:
    st.subheader("Pergerakan Sentimen Bulanan")
    st.caption(
        "Okt 2024 (mulai 20 Okt) dan Nov 2026 (sampai 9 Nov) adalah bulan parsial. "
        "Jumlah data tidak merata antar bulan, jadi baca grafik proporsi bersama grafik volume di bawahnya."
    )
    trend = df_f.groupby(["bulan", "sentiment"]).size().reset_index(name="jumlah")
    mode = st.radio("Tampilkan:", ["Proporsi (%)", "Jumlah"], horizontal=True)
    if mode == "Proporsi (%)":
        trend["jumlah"] = trend["jumlah"] / trend.groupby("bulan")["jumlah"].transform("sum") * 100
    st.plotly_chart(
        px.line(trend, x="bulan", y="jumlah", color="sentiment", markers=True,
                color_discrete_map=COLOR_MAP,
                labels={"bulan": "Bulan", "jumlah": mode, "sentiment": "Sentimen"}),
        width="stretch",
    )
    vol = df_f.groupby("bulan").size().reset_index(name="jumlah")
    st.plotly_chart(
        px.bar(vol, x="bulan", y="jumlah", title="Volume data per bulan",
               labels={"bulan": "Bulan", "jumlah": "Jumlah teks"}),
        width="stretch",
    )
    if st.checkbox("Tampilkan % negatif per sumber data"):
        ms = df_f.groupby(["bulan", "sumber_data"]).agg(
            n=("sentiment", "size"), neg=("sentiment", lambda s: (s == "Negatif").mean() * 100)
        ).reset_index()
        ms = ms[ms["n"] >= 20]  # bulan dengan data sangat sedikit disembunyikan
        st.plotly_chart(
            px.line(ms, x="bulan", y="neg", color="sumber_data", markers=True,
                    labels={"bulan": "Bulan", "neg": "% Negatif", "sumber_data": "Sumber"}),
            width="stretch",
        )

# ---------- TAB 3 ----------
with tab3:
    left, right = st.columns(2)
    with left:
        st.subheader("Proporsi Sentimen per Sumber")
        p = df_f.groupby(["sumber_data", "sentiment"]).size().reset_index(name="jumlah")
        p["persen"] = p["jumlah"] / p.groupby("sumber_data")["jumlah"].transform("sum") * 100
        st.plotly_chart(
            px.bar(p, x="persen", y="sumber_data", color="sentiment", orientation="h",
                   color_discrete_map=COLOR_MAP, category_orders={"sumber_data": SRC_ORDER},
                   labels={"persen": "Persen (%)", "sumber_data": "Sumber"}),
            width="stretch",
        )
    with right:
        st.subheader("Proporsi Sentimen per Kata Kunci")
        rows = []
        for k in kw_opt:
            sub = df_f[df_f[f"kw_{k}"]]
            for s, n in sub["sentiment"].value_counts().items():
                rows.append({"kata_kunci": k, "sentiment": s, "jumlah": n})
        if rows:
            k_df = pd.DataFrame(rows)
            k_df["persen"] = k_df["jumlah"] / k_df.groupby("kata_kunci")["jumlah"].transform("sum") * 100
            st.plotly_chart(
                px.bar(k_df, x="persen", y="kata_kunci", color="sentiment", orientation="h",
                       color_discrete_map=COLOR_MAP,
                       labels={"persen": "Persen (%)", "kata_kunci": "Kata kunci"}),
                width="stretch",
            )

# ---------- TAB 4 ----------
with tab4:
    st.subheader("Performa Model IndoBERT (data uji)")
    p_metrics = hasil("metrics.json")
    metrics = json.load(open(p_metrics)) if p_metrics else {}
    nama = [("accuracy", "Accuracy"), ("precision", "Precision (macro)"),
            ("recall", "Recall (macro)"), ("f1_macro", "Macro F1")]
    if any(metrics.get(k) is not None for k, _ in nama):
        for col, (k, label) in zip(st.columns(4), nama):
            v = metrics.get(k)
            col.metric(label, f"{v * 100:.2f}%" if v is not None else "-")
    else:
        st.info("Isi hasil/metrics.json dengan angka evaluasi dari Anggota 1.")

    cl, cr = st.columns(2)
    p_eval = hasil("tabel_evaluasi_indobert_160k.csv", "tabel_evaluasi_indobert.csv")
    if p_eval:
        cl.markdown("**Metrik per kelas**")
        cl.dataframe(pd.read_csv(p_eval, index_col=0), width="stretch")
    p_cm = hasil("confusion_matrix_indobert_160k.png", "confusion_matrix_indobert.png", "confusion_matrix.png")
    if p_cm:
        cr.image(p_cm, caption="Confusion Matrix IndoBERT")

    st.subheader("Uji Chi-Square")
    st.caption("Dengan data sebesar ini hampir semua uji signifikan (p < 0,001); baca ukuran efek Cramér's V.")
    for judul, f in [("Antar sumber data", "2b_chi_square_pasangan_sumber.csv"),
                     ("Antar kata kunci", "1b_chi_square_kata_kunci.csv"),
                     ("Antar periode (temporal)", "3_chi_square_temporal.csv")]:
        p = hasil(f)
        if p:
            st.markdown(f"**{judul}**")
            st.dataframe(pd.read_csv(p), width="stretch", hide_index=True)

    st.subheader("Topik Pemicu Sentimen Negatif")
    p_top = hasil("4_tabel_topik_negatif_semua.csv")
    if p_top:
        top = pd.read_csv(p_top)
        if top["nama_topik"].astype(str).str.contains("beri nama").any():
            st.warning("Nama topik belum diisi (masih '(beri nama)'). Isi kolom nama_topik di CSV dulu.")
        st.dataframe(top, width="stretch", hide_index=True)
    else:
        st.info("Taruh 4_tabel_topik_negatif_semua.csv (dari Anggota 2) di folder hasil/.")
    p_img = hasil("4_topik_negatif_kata_kunci_semua.png")
    if p_img:
        st.image(p_img, caption="Kata kunci tiap topik")

# ---------- TAB 5 ----------
with tab5:
    st.subheader("Intip Sampel Data")
    q = st.text_input("Cari kata di teks bersih:", "")
    tabel = df_f
    if q:
        tabel = df_f[df_f["full_text_clean"].str.contains(q, case=False, na=False, regex=False)]
    st.caption(f"{len(tabel):,} baris cocok (ditampilkan maks. 100)")
    st.dataframe(
        tabel[["doc_id", "sumber_data", "kata_kunci", "created_at", "full_text_clean", "sentiment"]].head(100),
        width="stretch",
    )