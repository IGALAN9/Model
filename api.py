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
# Batasi thread TensorFlow agar proses penjelasan antar-model dapat berbagi CPU.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(3)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, regularizers

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lokasi file dan parameter inferensi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_BILSTM   = os.path.join(BASE_DIR, "src", "models",    "best_model_bilstm.keras")
MODEL_GRU      = os.path.join(BASE_DIR, "src", "models",    "best_model_gru.keras")
MODEL_CNN      = os.path.join(BASE_DIR, "src", "models",    "best_model_cnn_bilstm.keras")

# Pemetaan kata pada tokenizer harus sesuai dengan tokenizer saat training.
TOKENIZER_BILSTM_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "tokenizer_bilstm.pkl")
TOKENIZER_SHARED_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "tokenizer_gru.pkl")  # Untuk GRU dan CNN.
TFIDF_PATH     = os.path.join(BASE_DIR, "src", "tokenizer", "tfidf_vectorizer.pkl")
BACKGROUND_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "background_samples.pkl")
BACKGROUND_SIZE = 50  # Jumlah teks acuan untuk penjelasan SHAP.
SHAP_NSAMPLES = 50  # Jumlah sampel estimasi SHAP per input.

# Panjang sequence harus sesuai dengan bentuk input model yang dimuat.
MAX_LEN_BILSTM = 200
MAX_LEN_GRU    = 250
MAX_LEN_CNN    = 250
THRESHOLD = 0.80  # Ambang rata-rata skor untuk label HOAKS.

# Layer attention
class AttentionLayer(layers.Layer):
    """Menggabungkan fitur sequence menggunakan bobot attention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Membuat bobot attention dan bias untuk setiap posisi sequence."""
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
        """Menghitung bobot attention dan menjumlahkan fitur sesuai bobotnya."""
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        """Menyediakan konfigurasi layer untuk penyimpanan dan pemuatan model."""
        return super().get_config()

# Resource aplikasi
app_state: dict = {}

CUSTOM_OBJECTS = {"AttentionLayer": AttentionLayer}
VOCAB_SIZE: dict[str, int] = {}


def _check_files() -> None:
    """Memastikan file model, tokenizer, dan TF-IDF tersedia sebelum startup."""
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
    """Mencari layer Embedding pertama, termasuk di dalam model bersarang."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            return layer
        if hasattr(layer, "layers"):
            found = _find_embedding_layer_recursive(layer)
            if found is not None:
                return found
    return None


def _embedding_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Mengambil layer Embedding atau melaporkan error jika tidak ditemukan."""
    layer = _find_embedding_layer_recursive(model)
    if layer is None:
        raise ValueError(f"Tidak ditemukan layer Embedding di model '{model.name}'.")
    return layer


def _get_embedding_vocab_size(model: tf.keras.Model) -> int:
    """Membaca ukuran vocabulary dari layer Embedding model."""
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
    """Memisahkan model menjadi pembentuk embedding dan prediksi dari embedding.

    SHAP memakai embedding float agar gradien dapat dihitung. extra_inputs
    mempertahankan input tambahan, seperti cabang TF-IDF pada model CNN.
    """
    embed_layer = _embedding_layer(model)
    embed_model = tf.keras.Model(inputs=embed_layer.input, outputs=embed_layer.output)

    post_inputs = [embed_layer.output] + (extra_inputs or [])
    post_embed_model = tf.keras.Model(inputs=post_inputs, outputs=model.output)

    return embed_model, post_embed_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat resource saat startup dan melepaskannya saat aplikasi berhenti."""
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

    app_state["tokenizers"] = {
        "bilstm": tokenizer_bilstm,
        "gru":    tokenizer_shared,
        "cnn":    tokenizer_shared,
    }

    with open(TFIDF_PATH, "rb") as f:
        app_state["tfidf"] = pickle.load(f)

    print("✅ Semua model & vectorizer berhasil dimuat.")

    # Tampilkan konfigurasi tokenizer dan embedding untuk pemeriksaan startup.
    for _name in ("bilstm", "gru", "cnn"):
        _tok = app_state["tokenizers"][_name]
        print(f"Tokenizer '{_name}' — num_words: {_tok.num_words} | word_index size: {len(_tok.word_index)}")
    print("VOCAB_SIZE per model (dari embedding layer):", VOCAB_SIZE)

    # Validasi ukuran vocabulary; kesamaan ukuran belum menjamin pemetaan kata sama.
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

    # Pemetaan balik indeks ke kata untuk menampilkan kontribusi SHAP.
    app_state["index_to_word"] = {
        name: {idx: word for word, idx in tok.word_index.items()}
        for name, tok in app_state["tokenizers"].items()
    }
    app_state["tfidf_feature_names"] = app_state["tfidf"].get_feature_names_out()

    app_state["explainers"] = {}
    app_state["embed_models"] = {}
    app_state["executor"] = ThreadPoolExecutor(max_workers=3)
    app_state["post_embed_models"] = {}
    # Background menjadi acuan perhitungan kontribusi SHAP.
    if os.path.exists(BACKGROUND_PATH):
        with open(BACKGROUND_PATH, "rb") as f:
            background_texts = pickle.load(f)[:BACKGROUND_SIZE]

        bg_bilstm = _pad(background_texts, app_state["tokenizers"]["bilstm"], MAX_LEN_BILSTM, VOCAB_SIZE["bilstm"])
        bg_gru    = _pad(background_texts, app_state["tokenizers"]["gru"],    MAX_LEN_GRU,    VOCAB_SIZE["gru"])
        bg_cnn    = _pad(background_texts, app_state["tokenizers"]["cnn"],    MAX_LEN_CNN,    VOCAB_SIZE["cnn"])
        bg_tfidf  = app_state["tfidf"].transform(background_texts).toarray().astype(np.float32)

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

        # Pertahankan input TF-IDF saat memisahkan cabang embedding CNN.
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

# Konfigurasi FastAPI
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

# Format request dan response
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
    score: float  # Positif menuju HOAKS, negatif menuju FAKTA, relatif terhadap background.


class ExplainResponse(BaseModel):
    text:        str
    label:       str
    confidence:  float
    word_scores: list[WordScore]  # Diurutkan berdasarkan nilai absolut kontribusi.


class VocabCheckRequest(BaseModel):
    words: Annotated[list[str], Field(min_length=1, max_length=50)]

    model_config = {
        "json_schema_extra": {
            "examples": [{"words": ["chip", "cip", "mengandung", "vaksin"]}]
        }
    }


class VocabCheckResult(BaseModel):
    word:          str
    word_lower:    str   # Bentuk lowercase yang digunakan untuk pencarian vocabulary.
    in_vocab_bilstm: bool
    in_vocab_gru_cnn: bool
    tokenizer_index_bilstm: int | None
    tokenizer_index_gru_cnn: int | None
    tfidf_index:     int | None


class VocabCheckResponse(BaseModel):
    results: list[VocabCheckResult]


class RawScoresResponse(BaseModel):
    text:             str
    p_bilstm:         float
    p_gru:            float
    p_cnn:            float
    avg:              float   # Rata-rata skor ketiga model.
    threshold:        float
    would_be_hoaks:   bool    # Bernilai True jika avg >= threshold.
    gap_to_threshold: float   # avg - threshold; negatif berarti di bawah ambang HOAKS.
    note: str = (
        "avg dihitung dgn np.mean([p_bilstm, p_gru, p_cnn]) — SAMA PERSIS dgn "
        "_build_response(). Kalau avg < threshold walau p_bilstm/p_gru/p_cnn "
        "semua > 0.5 (condong HOAKS), label akhir tetap FAKTA karena threshold "
        "0.80 belum tercapai — ini beda dari kasus modelnya salah baca teks."
    )

# Persiapan input dan prediksi
def _pad(texts: list[str], tokenizer, maxlen: int, vocab_size: int) -> np.ndarray:
    """Mengubah teks menjadi sequence dengan padding dan pembatasan indeks.

    Indeks yang terpotong dapat mewakili kata lain; label SHAP memakai
    _original_sequence agar tetap merujuk pada indeks tokenizer semula.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")
    return np.clip(padded, 0, vocab_size - 1)


def _original_sequence(texts: list[str], tokenizer, maxlen: int) -> np.ndarray:
    """Membuat sequence tanpa clipping untuk pemetaan indeks ke kata pada SHAP."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")


def _tfidf(texts: list[str]) -> np.ndarray:
    """Mengubah teks menjadi fitur TF-IDF float32 untuk model CNN hybrid."""
    return app_state["tfidf"].transform(texts).toarray().astype(np.float32)


def _predict_single(text: str) -> tuple[float, float, float]:
    """Menghasilkan skor HOAKS dari BiLSTM, GRU, dan CNN untuk satu teks."""
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
    """Menggabungkan skor model dan menghitung label serta confidence dari threshold."""
    avg     = float(np.mean([p_bilstm, p_gru, p_cnn]))

    is_fake = avg >= THRESHOLD

    # Confidence dipetakan ke rentang 0,5–1 berdasarkan posisi skor terhadap threshold.
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


# Penjelasan prediksi dengan SHAP
def _squeeze_shap_output(raw) -> np.ndarray:
    """Menghapus dimensi berukuran satu dari keluaran SHAP."""
    arr = np.array(raw)
    return arr.squeeze()


def _dup(arr: np.ndarray) -> np.ndarray:
    """Menduplikasi batch untuk SHAP; pemanggil memakai hasil sampel pertama."""
    return np.concatenate([arr, arr], axis=0)


def _sequence_shap_to_words(shap_vals_seq: np.ndarray, padded_seq: np.ndarray, index_to_word: dict) -> dict[str, float]:
    """Menjumlahkan kontribusi token per kata, tanpa padding dan token OOV."""
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
    """Mengambil kontribusi fitur TF-IDF yang bernilai nonzero pada input.

    Fitur yang tidak muncul diabaikan agar hasil relevan untuk highlight teks.
    """
    scores: dict[str, float] = {}
    for i, present in enumerate(tfidf_input_vec):
        if present == 0:
            continue
        word = feature_names[i]
        scores[word] = scores.get(word, 0.0) + float(shap_vals_tfidf[i])
    return scores

def _explain_bilstm(text: str) -> tuple[dict[str, float], float]:
    """Menghitung kontribusi kata dan skor prediksi dari model BiLSTM."""
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
    """Menghitung kontribusi kata dan skor prediksi dari model GRU."""
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
    """Menggabungkan kontribusi sequence dan TF-IDF serta skor prediksi CNN."""
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
    """Menjalankan penjelasan tiga model secara paralel dan menggabungkan skor kata."""
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
    # Setiap kata dirata-ratakan hanya dari model yang menyertakannya.
    merged_scores = {w: totals[w] / counts[w] for w in totals}

    return merged_scores, p_bilstm, p_gru, p_cnn


# Endpoint API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

@app.get("/", tags=["Health"])
def root():
    """Mengembalikan pesan bahwa API sedang berjalan."""
    return {"status": "ok", "message": "Fake News Detection API berjalan."}


@app.get("/health", tags=["Health"])
def health():
    """Melaporkan status pemuatan resource dari state aplikasi."""
    loaded = all(k in app_state for k in ["bilstm", "gru", "cnn", "tokenizer", "tfidf"])
    return {"status": "ok" if loaded else "error", "models_loaded": loaded}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Menerima satu teks dan mengembalikan hasil prediksi ensemble."""
    try:
        p_bilstm, p_gru, p_cnn = _predict_single(request.text)
        return _build_response(request.text, p_bilstm, p_gru, p_cnn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """Memproses maksimal 50 teks secara berurutan dan mengembalikan hasilnya."""
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
    """Mengembalikan prediksi dan kontribusi kata yang diurutkan menurut besarnya."""
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
    """Menampilkan skor tiap model, rata-rata, dan selisihnya terhadap threshold."""
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
    """Memeriksa kata di word_index kedua tokenizer dan vocabulary TF-IDF."""
    tokenizer_bilstm = app_state["tokenizers"]["bilstm"]
    tokenizer_gru_cnn = app_state["tokenizers"]["gru"]  # Tokenizer bersama GRU dan CNN.

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
