from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import pickle
import os
import shap  
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
# Batasi thread internal TF per operasi, supaya 3 model yang dijalankan
# paralel (lihat ThreadPoolExecutor di _explain_single) tidak saling
# rebutan seluruh core CPU — biar paralelisasi level model yang jalan,
# bukan diserap habis oleh intra-op parallelism TF sendiri.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(3)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, regularizers

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_BILSTM   = os.path.join(BASE_DIR, "src", "models",    "best_model_bilstm.keras")
MODEL_GRU      = os.path.join(BASE_DIR, "src", "models",    "best_model_gru.keras")
MODEL_CNN      = os.path.join(BASE_DIR, "src", "models",    "best_model_cnn_bilstm.keras")

# PENTING — root cause bias prediksi bilstm ke HOAKS: bilstm dilatih dengan
# tokenizer HASIL FIT SENDIRI (VOCAB_SIZE=50000, lihat notebook training),
# tapi gru & cnn dilatih dengan tokenizer fit terpisah (VOCAB_SIZE=20000).
# Kedua notebook menyimpan ke NAMA FILE YANG SAMA ("tokenizer.pkl"), sehingga
# salah satu menimpa yang lain di Google Drive — dan api.py lama cuma load
# SATU tokenizer.pkl dipakai bareng untuk ketiga model. Akibatnya word_index
# yang dipakai saat inferensi bilstm TIDAK COCOK dengan word_index yang dia
# lihat saat training, membuat setiap kata dipetakan ke index "asing" bagi
# bilstm — inilah yang menyebabkan prediksinya ngaco/bias ke HOAKS.
#
# Fix: bilstm sekarang punya file tokenizer sendiri (tokenizer_bilstm.pkl),
# terpisah dari tokenizer.pkl yang dipakai gru & cnn. tokenizer_bilstm.pkl
# HARUS di-refit dari X_train yang SAMA PERSIS dengan waktu bilstm dilatih
# (fit_on_texts bersifat deterministik terhadap corpus yang sama) — lihat
# instruksi refit di notebook model bilstm.
# PENTING (update ke-2): tokenizer.pkl yang lama TERBUKTI SUDAH TERTIMPA juga
# — hasil debug menunjukkan tokenizer.pkl yang "seharusnya" untuk gru/cnn
# (num_words=20000) ternyata sekarang num_words=50000, identik dengan
# tokenizer bilstm. Artinya tokenizer.pkl sudah lama tidak bisa dipercaya
# sebagai tokenizer asli gru/cnn. Pakai file baru yang eksplisit hasil refit
# ulang dari X_train notebook gru/cnn (num_words=20000).
TOKENIZER_BILSTM_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "tokenizer_bilstm.pkl")
TOKENIZER_SHARED_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "tokenizer_gru.pkl")  # dipakai gru & cnn
TFIDF_PATH     = os.path.join(BASE_DIR, "src", "tokenizer", "tfidf_vectorizer.pkl")
BACKGROUND_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "background_samples.pkl")
BACKGROUND_SIZE = 50  # jumlah sampel background yang dipakai (trade-off akurasi vs kecepatan)
SHAP_NSAMPLES = 50 # default shap = 200; makin kecil makin cepat, tapi skor makin noisy

MAX_LEN_BILSTM = 200
MAX_LEN_GRU    = 250
MAX_LEN_CNN    = 250
THRESHOLD = 0.80

# ─── Custom Attention Layer ───────────────────────────────────────────────────
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            regularizer=regularizers.l2(1e-4),
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# ─── App State ────────────────────────────────────────────────────────────────
app_state: dict = {}

CUSTOM_OBJECTS = {"AttentionLayer": AttentionLayer}
VOCAB_SIZE: dict[str, int] = {}


def _check_files() -> None:
    missing = [
        p for p in [MODEL_BILSTM, MODEL_GRU, MODEL_CNN, TOKENIZER_BILSTM_PATH, TOKENIZER_SHARED_PATH, TFIDF_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            "File tidak ditemukan:\n" + "\n".join(missing) +
            "\n\nKalau tokenizer_bilstm.pkl yang hilang: refit tokenizer di notebook "
            "bilstm dengan X_train yang SAMA PERSIS seperti waktu training, lalu "
            "simpan sebagai tokenizer_bilstm.pkl (JANGAN timpa tokenizer.pkl yang "
            "dipakai gru/cnn)."
        )


def _find_embedding_layer_recursive(model: tf.keras.Model) -> tf.keras.layers.Layer | None:
    """Cari layer Embedding PERTAMA di model, rekursif ke dalam submodel/layer
    bersarang (mis. model dibangun dengan blok Functional/Sequential di dalam
    Functional lain). Pencarian by-name saja ("embedding" in layer.name) TIDAK
    CUKUP kalau layer Embedding-nya bersarang di dalam submodel — dia tidak
    akan muncul di model.layers top-level sama sekali, dan pencarian gagal
    secara DIAM-DIAM (lihat _get_embedding_vocab_size lama yang jatuh ke
    fallback 50000 untuk bilstm, padahal vocab asli bilstm bukan 50000 —
    inilah akar penyebab bias prediksi bilstm ke HOAKS)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            return layer
        # Turun rekursif kalau layer ini sendiri adalah model/container bersarang
        if hasattr(layer, "layers"):
            found = _find_embedding_layer_recursive(layer)
            if found is not None:
                return found
    return None


def _embedding_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Cari layer Embedding pertama di model (rekursif). Dipakai untuk memotong
    model jadi dua bagian saat setup GradientExplainer (lihat _build_embedding_submodels)."""
    layer = _find_embedding_layer_recursive(model)
    if layer is None:
        raise ValueError(f"Tidak ditemukan layer Embedding di model '{model.name}'.")
    return layer


def _get_embedding_vocab_size(model: tf.keras.Model) -> int:
    """Ambil input_dim dari layer Embedding pertama di model (rekursif).
    SENGAJA raise error kalau tidak ketemu — JANGAN fallback diam-diam ke
    angka default, karena angka default yang salah akan lolos ke _pad() dan
    membuat clipping index salah tanpa error apa pun yang kelihatan (ini yang
    terjadi pada bilstm sebelumnya: fallback 50000 dipakai padahal vocab asli
    bilstm bukan 50000, menyebabkan lookup Embedding ke luar jangkauan asli
    dan prediksi bilstm jadi ngaco/bias ke HOAKS)."""
    layer = _find_embedding_layer_recursive(model)
    if layer is None:
        raise ValueError(
            f"Tidak ditemukan layer Embedding di model '{model.name}' — "
            "cek model.summary() untuk lihat struktur aslinya."
        )
    return layer.get_config()["input_dim"]


def _build_embedding_submodels(
    model: tf.keras.Model, extra_inputs: list | None = None
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    ROOT CAUSE FIX untuk error "zero-dimensional arrays cannot be concatenated":
    GradientExplainer butuh gradien output model terhadap INPUT model. Tapi input
    model kita adalah index integer hasil tokenizer — index integer tidak punya
    gradien (cuma dipakai untuk lookup di layer Embedding), jadi tf.GradientTape
    mengembalikan None, lalu shap mencoba np.concatenate(None, ...) dan meledak.
    Ini bug/limitasi shap yang sudah lama dilaporkan (shap issue #965, #496,
    #1119) dan TIDAK bisa diperbaiki dengan gonta-ganti versi numpy/shap.

    Fix-nya: jangan explain dari input integer, tapi dari OUTPUT layer Embedding
    (berupa vektor float, punya gradien). Makanya model dipecah jadi dua:
      - embed_model      : input token index (int)      -> vektor embedding (float)
      - post_embed_model : vektor embedding (+extra_inputs) -> prediksi akhir
    GradientExplainer dijalankan di post_embed_model, dengan background data
    berupa hasil embed_model.predict(...), bukan sequence mentah.

    extra_inputs: tensor input lain yang ikut dipertahankan apa adanya di
    post_embed_model (dipakai untuk cabang TF-IDF di model CNN hybrid, yang
    memang sudah berupa angka float sehingga punya gradien dan tidak perlu
    "dipotong" seperti cabang sequence).
    """
    embed_layer = _embedding_layer(model)
    embed_model = tf.keras.Model(inputs=embed_layer.input, outputs=embed_layer.output)

    post_inputs = [embed_layer.output] + (extra_inputs or [])
    post_embed_model = tf.keras.Model(inputs=post_inputs, outputs=model.output)

    return embed_model, post_embed_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_files()

    app_state["bilstm"] = tf.keras.models.load_model(
        MODEL_BILSTM, custom_objects=CUSTOM_OBJECTS, compile=False
    )
    app_state["gru"] = tf.keras.models.load_model(
        MODEL_GRU, custom_objects=CUSTOM_OBJECTS, compile=False
    )
    app_state["cnn"] = tf.keras.models.load_model(
        MODEL_CNN, custom_objects=CUSTOM_OBJECTS, compile=False
    )

    VOCAB_SIZE["bilstm"] = _get_embedding_vocab_size(app_state["bilstm"])
    VOCAB_SIZE["gru"]    = _get_embedding_vocab_size(app_state["gru"])
    VOCAB_SIZE["cnn"]    = _get_embedding_vocab_size(app_state["cnn"])
    print(f"📐 Vocab sizes — bilstm:{VOCAB_SIZE['bilstm']} | gru:{VOCAB_SIZE['gru']} | cnn:{VOCAB_SIZE['cnn']}")

    with open(TOKENIZER_BILSTM_PATH, "rb") as f:
        tokenizer_bilstm = pickle.load(f)
    with open(TOKENIZER_SHARED_PATH, "rb") as f:
        tokenizer_shared = pickle.load(f)

    # ROOT CAUSE FIX bias bilstm ke HOAKS: dulu ketiga model berbagi SATU
    # tokenizer.pkl, padahal bilstm dilatih dengan tokenizer hasil fit
    # terpisah (VOCAB_SIZE=50000) yang tertimpa oleh tokenizer gru/cnn
    # (VOCAB_SIZE=20000) karena disimpan ke nama file yang sama saat training.
    # Sekarang tiap model pakai tokenizer yang benar-benar cocok dengan
    # word_index yang dia lihat saat training.
    app_state["tokenizers"] = {
        "bilstm": tokenizer_bilstm,
        "gru":    tokenizer_shared,
        "cnn":    tokenizer_shared,
    }

    with open(TFIDF_PATH, "rb") as f:
        app_state["tfidf"] = pickle.load(f)

    print("✅ Semua model & vectorizer berhasil dimuat.")

    # ── DEBUG: pastikan tiap tokenizer benar & vocab_size-nya berpasangan ──
    for _name in ("bilstm", "gru", "cnn"):
        _tok = app_state["tokenizers"][_name]
        print(f"Tokenizer '{_name}' — num_words: {_tok.num_words} | word_index size: {len(_tok.word_index)}")
    print("VOCAB_SIZE per model (dari embedding layer):", VOCAB_SIZE)

    # VALIDASI OTOMATIS: num_words tokenizer HARUS SAMA dengan input_dim
    # embedding model. Kalau beda, berarti tokenizer yang ke-load bukan
    # tokenizer asli model itu (skenario "file tertimpa" yang dua kali
    # kejadian di project ini) — mending gagal keras (fail loud) saat startup
    # daripada diam-diam menghasilkan prediksi yang salah.
    for _name in ("bilstm", "gru", "cnn"):
        _tok_num_words = app_state["tokenizers"][_name].num_words
        _model_vocab   = VOCAB_SIZE[_name]
        if _tok_num_words != _model_vocab:
            raise RuntimeError(
                f"MISMATCH tokenizer vs model untuk '{_name}': "
                f"tokenizer.num_words={_tok_num_words} tapi embedding model input_dim={_model_vocab}. "
                "Tokenizer yang ke-load kemungkinan BUKAN tokenizer asli model ini "
                "(kemungkinan besar file tertimpa training model lain). "
                "Cek ulang file tokenizer yang dipakai untuk model ini sebelum lanjut."
            )
    print("✅ Validasi tokenizer vs model OK — num_words tiap tokenizer cocok dengan embedding model-nya.")
    # ── akhir debug ──

    # ── OPSI 1: Setup GradientExplainer per model ──────────────────────────
    # index_to_word sekarang HARUS per tokenizer, karena bilstm punya
    # word_index yang berbeda dari gru/cnn (lihat catatan di atas).
    app_state["index_to_word"] = {
        name: {idx: word for word, idx in tok.word_index.items()}
        for name, tok in app_state["tokenizers"].items()
    }
    app_state["tfidf_feature_names"] = app_state["tfidf"].get_feature_names_out()

    app_state["explainers"] = {}
    app_state["embed_models"] = {}
    app_state["executor"] = ThreadPoolExecutor(max_workers=3)
    app_state["post_embed_models"] = {}
    if os.path.exists(BACKGROUND_PATH):
        with open(BACKGROUND_PATH, "rb") as f:
            background_texts = pickle.load(f)[:BACKGROUND_SIZE]

        bg_bilstm = _pad(background_texts, app_state["tokenizers"]["bilstm"], MAX_LEN_BILSTM, VOCAB_SIZE["bilstm"])
        bg_gru    = _pad(background_texts, app_state["tokenizers"]["gru"],    MAX_LEN_GRU,    VOCAB_SIZE["gru"])
        bg_cnn    = _pad(background_texts, app_state["tokenizers"]["cnn"],    MAX_LEN_CNN,    VOCAB_SIZE["cnn"])
        bg_tfidf  = app_state["tfidf"].transform(background_texts).toarray().astype(np.float32)

        # OPSI 1 — GradientExplainer dijalankan di post_embed_model (lihat
        # _build_embedding_submodels), BUKAN di model asli. Ini fix untuk error
        # "zero-dimensional arrays cannot be concatenated" yang akar masalahnya
        # adalah gradien None dari layer Embedding terhadap input integer
        # (shap issue #965) — bukan soal versi numpy/shap yang bentrok.

        embed_bilstm, post_bilstm = _build_embedding_submodels(app_state["bilstm"])
        bg_embed_bilstm = embed_bilstm.predict(bg_bilstm, verbose=0)
        app_state["explainers"]["bilstm"] = shap.GradientExplainer(post_bilstm, bg_embed_bilstm)
        app_state["embed_models"]["bilstm"] = embed_bilstm
        app_state["post_embed_models"]["bilstm"] = post_bilstm

        embed_gru, post_gru = _build_embedding_submodels(app_state["gru"])
        bg_embed_gru = embed_gru.predict(bg_gru, verbose=0)
        app_state["explainers"]["gru"] = shap.GradientExplainer(post_gru, bg_embed_gru)
        app_state["embed_models"]["gru"] = embed_gru
        app_state["post_embed_models"]["gru"] = post_gru

        # Model CNN hybrid: cabang TF-IDF sudah berupa angka float (punya
        # gradien), jadi cukup dipertahankan sebagai extra_inputs apa adanya —
        # yang perlu "dipotong" cuma cabang sequence-nya.
        # CATATAN: nama layer Input untuk cabang TF-IDF diasumsikan "input_tfidf"
        # (sesuai key dict yang dipakai di _predict_single/model.predict()).
        # Kalau nama layer aslinya beda, sesuaikan string di get_layer() di bawah
        # — cek dengan app_state["cnn"].summary() kalau terjadi KeyError di sini.
        tfidf_input_tensor = app_state["cnn"].get_layer("input_tfidf").output
        embed_cnn, post_cnn = _build_embedding_submodels(
            app_state["cnn"], extra_inputs=[tfidf_input_tensor]
        )
        bg_embed_cnn = embed_cnn.predict(bg_cnn, verbose=0)
        app_state["explainers"]["cnn"] = shap.GradientExplainer(post_cnn, [bg_embed_cnn, bg_tfidf])
        app_state["embed_models"]["cnn"] = embed_cnn
        app_state["post_embed_models"]["cnn"] = post_cnn

        print(f"🔍 GradientExplainer siap (background: {len(background_texts)} sampel).")
    else:
        print(f"⚠️  Background samples tidak ditemukan di {BACKGROUND_PATH} — /predict/explain nonaktif.")

    yield
    app_state["executor"].shutdown(wait=True)
    app_state.clear()
    VOCAB_SIZE.clear()

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detection API",
    description=(
        "Deteksi berita palsu menggunakan ensemble 3 model: "
        "BiLSTM + Attention, GRU + Attention, dan CNN-BiLSTM + TF-IDF. "
        "Confidence score adalah rata-rata output sigmoid ketiga model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=1)]

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Scientists discover new treatment for cancer."}]
        }
    }


class ModelScores(BaseModel):
    bilstm:     float
    gru:        float
    cnn_bilstm: float


class PredictResponse(BaseModel):
    text:         str
    label:        str   
    confidence:   float 
    is_fake:      bool
    model_scores: ModelScores


class BatchPredictRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=50)]

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": ["Breaking: aliens land in Jakarta!", "WHO releases new guidelines."]}]
        }
    }


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class WordScore(BaseModel):
    word:  str
    score: float  # positif = mendorong ke arah HOAKS, negatif = ke arah FAKTA


class ExplainResponse(BaseModel):
    text:        str
    label:       str
    confidence:  float
    word_scores: list[WordScore]  # sudah digabung dari ketiga model, diurutkan dari kontribusi terbesar


class VocabCheckRequest(BaseModel):
    words: Annotated[list[str], Field(min_length=1, max_length=50)]

    model_config = {
        "json_schema_extra": {
            "examples": [{"words": ["chip", "cip", "mengandung", "vaksin"]}]
        }
    }


class VocabCheckResult(BaseModel):
    word:          str
    word_lower:    str   # tokenizer Keras biasanya lowercase semua kata sebelum dicek
    in_vocab_bilstm: bool
    in_vocab_gru_cnn: bool
    tokenizer_index_bilstm: int | None   # index di tokenizer_bilstm.pkl, None kalau OOV
    tokenizer_index_gru_cnn: int | None  # index di tokenizer.pkl (dipakai gru & cnn), None kalau OOV
    tfidf_index:     int | None  # None kalau kata ini tidak ada di vocabulary TF-IDF


class VocabCheckResponse(BaseModel):
    results: list[VocabCheckResult]


class RawScoresResponse(BaseModel):
    text:             str
    p_bilstm:         float
    p_gru:            float
    p_cnn:            float
    avg:              float   # rata-rata ketiga model, ini yang dibandingkan ke THRESHOLD
    threshold:        float
    would_be_hoaks:   bool    # avg >= threshold (logika sama persis dgn _build_response)
    gap_to_threshold: float   # avg - threshold; negatif = masih di bawah ambang HOAKS
    note: str = (
        "avg dihitung dgn np.mean([p_bilstm, p_gru, p_cnn]) — SAMA PERSIS dgn "
        "_build_response(). Kalau avg < threshold walau p_bilstm/p_gru/p_cnn "
        "semua > 0.5 (condong HOAKS), label akhir tetap FAKTA karena threshold "
        "0.80 belum tercapai — ini beda dari kasus modelnya salah baca teks."
    )

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _pad(texts: list[str], tokenizer, maxlen: int, vocab_size: int) -> np.ndarray:
    """Tokenize (pakai tokenizer MILIK MODEL yang bersangkutan — lihat
    app_state["tokenizers"], bilstm punya tokenizer beda dari gru/cnn), pad,
    lalu clip index agar tidak melebihi vocab embedding.
    PENTING: hasil clip ini cuma boleh dipakai untuk FEED ke model (embedding
    lookup), JANGAN dipakai untuk translate index->kata (lihat _original_sequence)
    — kalau index tokenizer aslinya > vocab_size-1, clip akan menggantinya jadi
    index kata LAIN yang kebetulan menempati vocab_size-1, bukan kata aslinya."""
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")
    return np.clip(padded, 0, vocab_size - 1)


def _original_sequence(texts: list[str], tokenizer, maxlen: int) -> np.ndarray:
    """Sequence tokenizer (MILIK MODEL yang bersangkutan) TANPA clip vocab_size
    — dipakai KHUSUS untuk translate index->kata yang benar di /predict/explain.
    Model tetap di-feed pakai versi ter-clip dari _pad(), tapi label kata di
    response harus pakai index asli ini, supaya kata yang index-nya kebetulan
    > vocab_size model tidak salah nyasar ke kata lain saat ditranslate
    index_to_word."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")


def _tfidf(texts: list[str]) -> np.ndarray:
    return app_state["tfidf"].transform(texts).toarray().astype(np.float32)


def _predict_single(text: str) -> tuple[float, float, float]:
    """Prediksi satu teks, kembalikan (prob_bilstm, prob_gru, prob_cnn)."""
    tokenizers = app_state["tokenizers"]
    pad_bilstm = _pad([text], tokenizers["bilstm"], MAX_LEN_BILSTM, VOCAB_SIZE["bilstm"])
    pad_gru    = _pad([text], tokenizers["gru"],    MAX_LEN_GRU,    VOCAB_SIZE["gru"])
    pad_cnn    = _pad([text], tokenizers["cnn"],    MAX_LEN_CNN,    VOCAB_SIZE["cnn"])
    tfidf_feat = _tfidf([text])

    p_bilstm = float(app_state["bilstm"].predict(pad_bilstm, verbose=0).flatten()[0])
    p_gru    = float(app_state["gru"].predict(pad_gru,       verbose=0).flatten()[0])
    p_cnn    = float(app_state["cnn"].predict(
        {"input_sequence": pad_cnn, "input_tfidf": tfidf_feat}, verbose=0
    ).flatten()[0])

    return p_bilstm, p_gru, p_cnn


def _build_response(text: str, p_bilstm: float, p_gru: float, p_cnn: float) -> PredictResponse:
    avg     = float(np.mean([p_bilstm, p_gru, p_cnn]))

    is_fake = avg >= THRESHOLD

    if is_fake:
        final_confidence = 0.5 + ((avg - THRESHOLD) / (1.0 - THRESHOLD)) * 0.5
    else:
        final_confidence = 1.0 - (avg / THRESHOLD) * 0.5

    return PredictResponse(
        text=text,
        label="HOAKS" if is_fake else "FAKTA",
        confidence=round(final_confidence, 4), 
        is_fake=is_fake,
        model_scores=ModelScores(
            bilstm=round(p_bilstm, 4),
            gru=round(p_gru, 4),
            cnn_bilstm=round(p_cnn, 4),
        ),
    )


# ─── OPSI 1: GradientExplainer helpers ─────────────────────────────────────────
def _squeeze_shap_output(raw) -> np.ndarray:
    """Normalisasi bentuk output shap_values() (beda-beda antar versi shap) jadi array 1D."""
    arr = np.array(raw)
    return arr.squeeze()


def _dup(arr: np.ndarray) -> np.ndarray:
    """Workaround: sebagian versi shap error kalau dikasih batch size 1. Duplikasi jadi 2, ambil baris pertama nanti."""
    return np.concatenate([arr, arr], axis=0)


def _sequence_shap_to_words(shap_vals_seq: np.ndarray, padded_seq: np.ndarray, index_to_word: dict) -> dict[str, float]:
    """Translate skor SHAP dari index tokenizer balik ke kata asli (index 0 = padding, dilewati)."""
    scores: dict[str, float] = {}
    seq = padded_seq.flatten()
    for token_index, score in zip(seq, shap_vals_seq):
        if token_index == 0:
            continue
        word = index_to_word.get(int(token_index))
        if word is None or word == "<OOV>":
            continue
        scores[word] = scores.get(word, 0.0) + float(score)
    return scores


def _tfidf_shap_to_words(
    shap_vals_tfidf: np.ndarray, tfidf_input_vec: np.ndarray, feature_names: np.ndarray
) -> dict[str, float]:
    """Translate skor SHAP dari kolom TF-IDF (khusus jalur hybrid CNN) balik ke
    kata asli. HANYA kata yang benar-benar muncul di teks input (nilai TF-IDF
    != 0) yang disertakan.

    PENTING: filter-nya berdasarkan nilai TF-IDF INPUT (tfidf_input_vec), BUKAN
    berdasarkan apakah skor SHAP-nya 0. SHAP menghitung kontribusi dibanding
    rata-rata background, jadi kata yang TIDAK ADA di teks (tfidf=0) tetap bisa
    dapat skor SHAP kecil non-zero (mencerminkan "ketiadaan kata ini dibanding
    artikel rata-rata"). Itu valid secara matematis tapi tidak relevan untuk
    highlight kata di artikel — makanya harus difilter dari sisi input, bukan
    dari sisi skor.
    """
    scores: dict[str, float] = {}
    for i, present in enumerate(tfidf_input_vec):
        if present == 0:
            continue
        word = feature_names[i]
        scores[word] = scores.get(word, 0.0) + float(shap_vals_tfidf[i])
    return scores

def _explain_bilstm(text: str) -> tuple[dict[str, float], float]:
    tokenizers        = app_state["tokenizers"]
    embed_models       = app_state["embed_models"]
    post_embed_models  = app_state["post_embed_models"]
    index_to_word      = app_state["index_to_word"]

    pad_bilstm  = _pad([text], tokenizers["bilstm"], MAX_LEN_BILSTM, VOCAB_SIZE["bilstm"])
    orig_bilstm = _original_sequence([text], tokenizers["bilstm"], MAX_LEN_BILSTM)

    emb_bilstm = embed_models["bilstm"].predict(pad_bilstm, verbose=0)
    sv_bilstm = _squeeze_shap_output(
        app_state["explainers"]["bilstm"].shap_values(_dup(emb_bilstm), nsamples=SHAP_NSAMPLES)
    )[0]
    sv_bilstm = sv_bilstm.sum(axis=-1)
    words_bilstm = _sequence_shap_to_words(sv_bilstm, orig_bilstm, index_to_word["bilstm"])
    p_bilstm = float(post_embed_models["bilstm"].predict(emb_bilstm, verbose=0).flatten()[0])
    return words_bilstm, p_bilstm


def _explain_gru(text: str) -> tuple[dict[str, float], float]:
    tokenizers        = app_state["tokenizers"]
    embed_models       = app_state["embed_models"]
    post_embed_models  = app_state["post_embed_models"]
    index_to_word      = app_state["index_to_word"]

    pad_gru  = _pad([text], tokenizers["gru"], MAX_LEN_GRU, VOCAB_SIZE["gru"])
    orig_gru = _original_sequence([text], tokenizers["gru"], MAX_LEN_GRU)

    emb_gru = embed_models["gru"].predict(pad_gru, verbose=0)
    sv_gru = _squeeze_shap_output(
        app_state["explainers"]["gru"].shap_values(_dup(emb_gru), nsamples=SHAP_NSAMPLES)
    )[0]
    sv_gru = sv_gru.sum(axis=-1)
    words_gru = _sequence_shap_to_words(sv_gru, orig_gru, index_to_word["gru"])
    p_gru = float(post_embed_models["gru"].predict(emb_gru, verbose=0).flatten()[0])
    return words_gru, p_gru


def _explain_cnn(text: str) -> tuple[dict[str, float], float]:
    tokenizers        = app_state["tokenizers"]
    embed_models       = app_state["embed_models"]
    post_embed_models  = app_state["post_embed_models"]
    index_to_word      = app_state["index_to_word"]

    pad_cnn    = _pad([text], tokenizers["cnn"], MAX_LEN_CNN, VOCAB_SIZE["cnn"])
    orig_cnn   = _original_sequence([text], tokenizers["cnn"], MAX_LEN_CNN)
    tfidf_feat = _tfidf([text])

    emb_cnn = embed_models["cnn"].predict(pad_cnn, verbose=0)
    sv_cnn_seq, sv_cnn_tfidf = app_state["explainers"]["cnn"].shap_values(
        [_dup(emb_cnn), _dup(tfidf_feat)], nsamples=SHAP_NSAMPLES
    )
    sv_cnn_seq   = _squeeze_shap_output(sv_cnn_seq)[0].sum(axis=-1)
    sv_cnn_tfidf = _squeeze_shap_output(sv_cnn_tfidf)[0]

    words_cnn_seq   = _sequence_shap_to_words(sv_cnn_seq, orig_cnn, index_to_word["cnn"])
    words_cnn_tfidf = _tfidf_shap_to_words(sv_cnn_tfidf, tfidf_feat.flatten(), app_state["tfidf_feature_names"])

    words_cnn: dict[str, float] = defaultdict(float)
    for w, s in words_cnn_seq.items():
        words_cnn[w] += s
    for w, s in words_cnn_tfidf.items():
        words_cnn[w] += s

    p_cnn = float(post_embed_models["cnn"].predict([emb_cnn, tfidf_feat], verbose=0).flatten()[0])
    return dict(words_cnn), p_cnn

def _explain_single(text: str) -> tuple[dict[str, float], float, float, float]:
    if not app_state["explainers"]:
        raise RuntimeError("GradientExplainer belum siap — background_samples.pkl tidak ditemukan saat startup.")

    executor = app_state["executor"]
    future_bilstm = executor.submit(_explain_bilstm, text)
    future_gru    = executor.submit(_explain_gru, text)
    future_cnn    = executor.submit(_explain_cnn, text)

    words_bilstm, p_bilstm = future_bilstm.result()
    words_gru, p_gru       = future_gru.result()
    words_cnn, p_cnn       = future_cnn.result()

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for words in (words_bilstm, words_gru, words_cnn):
        for w, s in words.items():
            totals[w] += s
            counts[w] += 1
    merged_scores = {w: totals[w] / counts[w] for w in totals}

    return merged_scores, p_bilstm, p_gru, p_cnn


# ─── Routes ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Fake News Detection API berjalan."}


@app.get("/health", tags=["Health"])
def health():
    loaded = all(k in app_state for k in ["bilstm", "gru", "cnn", "tokenizer", "tfidf"])
    return {"status": "ok" if loaded else "error", "models_loaded": loaded}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Prediksi satu teks: ensemble 3 model, confidence = rata-rata sigmoid."""
    try:
        p_bilstm, p_gru, p_cnn = _predict_single(request.text)
        return _build_response(request.text, p_bilstm, p_gru, p_cnn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """Prediksi beberapa teks sekaligus (maks. 50): ensemble 3 model."""
    try:
        results = [
            _build_response(text, *_predict_single(text))
            for text in request.texts
        ]
        return BatchPredictResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain", response_model=ExplainResponse, tags=["Explainability"])
def predict_explain(request: PredictRequest):
    """
    OPSI 1 — Prediksi + penjelasan XAI via GradientExplainer per model.
    Menjalankan SHAP GradientExplainer terpisah untuk bilstm, gru, dan cnn,
    lalu menggabungkan skor kontribusi kata dari ketiganya. Lebih berat
    secara komputasi dibanding /predict biasa — cocok dipakai saat pengguna
    memang minta penjelasan, bukan dipanggil di setiap prediksi.
    """
    if not app_state.get("explainers"):
        raise HTTPException(
            status_code=503,
            detail="Explainability belum aktif — background_samples.pkl tidak ditemukan saat startup.",
        )
    try:
        merged_scores, p_bilstm, p_gru, p_cnn = _explain_single(request.text)
        base_response = _build_response(request.text, p_bilstm, p_gru, p_cnn)

        word_scores = sorted(
            (WordScore(word=w, score=round(s, 4)) for w, s in merged_scores.items()),
            key=lambda ws: abs(ws.score),
            reverse=True,
        )

        return ExplainResponse(
            text=request.text,
            label=base_response.label,
            confidence=base_response.confidence,
            word_scores=word_scores,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/raw-scores", response_model=RawScoresResponse, tags=["Debug"])
def raw_scores(request: PredictRequest):
    """
    Kembalikan probabilitas MENTAH dari ketiga model (bilstm, gru, cnn) TANPA
    melewati THRESHOLD final — supaya kelihatan beda antara dua kasus yang
    gejalanya sama-sama "kelabel FAKTA" tapi akar masalahnya beda:
      1) Model memang salah baca teks (avg rendah, jauh di bawah 0.5) →
         indikasi model tidak mengenali pola/kata dalam teks tsb.
      2) Model sebenarnya condong ke HOAKS (avg > 0.5) tapi belum tembus
         THRESHOLD 0.80 → bukan model yang "buta", tapi ambang batasnya yang
         ketat. Kalau ini yang terjadi, evaluasi ulang apakah THRESHOLD=0.80
         memang disengaja atau perlu diturunkan.
    """
    try:
        p_bilstm, p_gru, p_cnn = _predict_single(request.text)
        avg = float(np.mean([p_bilstm, p_gru, p_cnn]))
        return RawScoresResponse(
            text=request.text,
            p_bilstm=round(p_bilstm, 4),
            p_gru=round(p_gru, 4),
            p_cnn=round(p_cnn, 4),
            avg=round(avg, 4),
            threshold=THRESHOLD,
            would_be_hoaks=avg >= THRESHOLD,
            gap_to_threshold=round(avg - THRESHOLD, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/vocab-check", response_model=VocabCheckResponse, tags=["Debug"])
def vocab_check(request: VocabCheckRequest):
    """
    Cek apakah suatu kata dikenal (ada di vocabulary) tokenizer sequence
    dan/atau TF-IDF vectorizer, atau malah OOV (out-of-vocabulary).
    Berguna untuk investigasi kenapa suatu kata dapat skor SHAP yang
    janggal/tidak sesuai intuisi — kalau in_vocab=False, kata itu tidak
    pernah "dilihat" model dengan bentuk aslinya saat training.

    Menampilkan hasil untuk KEDUA tokenizer terpisah (bilstm vs gru/cnn),
    supaya kelihatan langsung kalau index-nya berbeda antar model — ini yang
    dulu jadi akar penyebab bias prediksi bilstm ke HOAKS.
    """
    tokenizer_bilstm = app_state["tokenizers"]["bilstm"]
    tokenizer_gru_cnn = app_state["tokenizers"]["gru"]  # sama dgn ["cnn"]

    results = []
    for word in request.words:
        word_lower = word.lower()
        tok_idx_bilstm = tokenizer_bilstm.word_index.get(word_lower)
        tok_idx_gru_cnn = tokenizer_gru_cnn.word_index.get(word_lower)
        tfidf_idx = app_state["tfidf"].vocabulary_.get(word_lower)

        results.append(VocabCheckResult(
            word=word,
            word_lower=word_lower,
            in_vocab_bilstm=tok_idx_bilstm is not None,
            in_vocab_gru_cnn=tok_idx_gru_cnn is not None,
            tokenizer_index_bilstm=tok_idx_bilstm,
            tokenizer_index_gru_cnn=tok_idx_gru_cnn,
            tfidf_index=tfidf_idx,
        ))

    return VocabCheckResponse(results=results)