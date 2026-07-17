from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import pickle
import os
import shap  # OPSI 3: wrapper black-box (lihat lifespan & _explain_single)
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
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
TOKENIZER_PATH = os.path.join(BASE_DIR, "src", "tokenizer", "tokenizer.pkl")
TFIDF_PATH     = os.path.join(BASE_DIR, "src", "tokenizer", "tfidf_vectorizer.pkl")

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
        p for p in [MODEL_BILSTM, MODEL_GRU, MODEL_CNN, TOKENIZER_PATH, TFIDF_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError("File tidak ditemukan:\n" + "\n".join(missing))


def _get_embedding_vocab_size(model: tf.keras.Model) -> int:
    """Ambil input_dim dari layer Embedding pertama di model."""
    for layer in model.layers:
        if "embedding" in layer.name.lower():
            return layer.get_config()["input_dim"]
    return 50000  


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

    with open(TOKENIZER_PATH, "rb") as f:
        app_state["tokenizer"] = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        app_state["tfidf"] = pickle.load(f)

    print("✅ Semua model & vectorizer berhasil dimuat.")

    # ── PIVOT KE OPSI 3: wrapper black-box (bukan lagi DeepExplainer/GradientExplainer) ──
    # Setelah 2x percobaan (DeepExplainer, lalu GradientExplainer) tetap gagal
    # karena ketidakcocokan gradien custom AttentionLayer dengan TF2 di environment
    # ini, pindah ke pendekatan yang SAMA SEKALI TIDAK menyentuh gradien model.
    # shap.Explainer dengan Text masker cukup memanggil model.predict() berkali-kali
    # (persis seperti endpoint /predict yang sudah terbukti jalan normal), jadi
    # kelas bug gradien ini otomatis tidak relevan lagi.
    #
    # Bonus: karena caranya cuma manggil predict() pada teks yang diubah-ubah
    # (bukan berdasarkan baseline numerik), background_samples.pkl JADI TIDAK
    # DIPERLUKAN lagi untuk pendekatan ini.
    def _ensemble_predict_fn(texts: list[str]) -> np.ndarray:
        """Fungsi predict() gabungan ensemble, dipakai sebagai 'kotak hitam' oleh SHAP."""
        probs = []
        for t in texts:
            p_bilstm, p_gru, p_cnn = _predict_single(t)
            probs.append(np.mean([p_bilstm, p_gru, p_cnn]))
        return np.array(probs)

    app_state["explain_fn"] = _ensemble_predict_fn
    # Text masker default SHAP memecah kata pakai regex yang menganggap tanda
    # hubung (-) sebagai pemisah — bikin "COVID-19" jadi "COVID-"+"19",
    # "diam-diam" jadi "diam"+"diam-". Pakai tokenizer regex \S+ (pisah
    # cuma berdasarkan spasi) supaya kata dengan tanda hubung tetap utuh.
    app_state["explainer"] = shap.Explainer(
        _ensemble_predict_fn, shap.maskers.Text(r"\S+")
    )
    print("🔍 SHAP wrapper explainer siap (black-box, tanpa gradien).")

    yield
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
    in_vocab:      bool
    tokenizer_index: int | None  # None kalau OOV (tidak dikenal tokenizer)
    tfidf_index:     int | None  # None kalau kata ini tidak ada di vocabulary TF-IDF


class VocabCheckResponse(BaseModel):
    results: list[VocabCheckResult]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _pad(texts: list[str], maxlen: int, vocab_size: int) -> np.ndarray:
    """Tokenize, pad, lalu clip index agar tidak melebihi vocab embedding."""
    sequences = app_state["tokenizer"].texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")
    return np.clip(padded, 0, vocab_size - 1)


def _tfidf(texts: list[str]) -> np.ndarray:
    return app_state["tfidf"].transform(texts).toarray().astype(np.float32)


def _predict_single(text: str) -> tuple[float, float, float]:
    """Prediksi satu teks, kembalikan (prob_bilstm, prob_gru, prob_cnn)."""
    pad_bilstm = _pad([text], MAX_LEN_BILSTM, VOCAB_SIZE["bilstm"])
    pad_gru    = _pad([text], MAX_LEN_GRU,    VOCAB_SIZE["gru"])
    pad_cnn    = _pad([text], MAX_LEN_CNN,    VOCAB_SIZE["cnn"])
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


# ─── OPSI 3: wrapper black-box helper ──────────────────────────────────────────
def _explain_single(text: str) -> tuple[dict[str, float], float, float, float]:
    """
    Jalankan SHAP wrapper (black-box) pada satu teks. Berbeda dari pendekatan
    gradien sebelumnya, di sini SHAP tidak pernah menyentuh index tokenizer,
    embedding, atau gradien model sama sekali — dia cuma memanggil
    _ensemble_predict_fn(list_teks) berkali-kali dengan variasi kata yang
    dihapus/di-mask, lalu membandingkan perubahan probabilitasnya. Hasilnya
    otomatis berupa skor PER KATA (bukan per index), jadi tidak perlu proses
    translate atau gabung-3-model manual seperti pendekatan sebelumnya.
    """
    explainer = app_state.get("explainer")
    if explainer is None:
        raise RuntimeError("SHAP explainer belum siap.")

    explanation = explainer([text], max_evals=500)  # naikkan dari default supaya skor lebih stabil

    words = explanation.data[0]     # array kata hasil pemecahan teks oleh Text masker
    scores = explanation.values[0]  # array skor SHAP, urutannya sejajar dengan `words`

    merged_scores: dict[str, float] = defaultdict(float)
    for word, score in zip(words, scores):
        w = str(word).strip()
        if not w:
            continue
        merged_scores[w] += float(score)

    # Prediksi label tetap pakai jalur predict biasa (bukan dari SHAP),
    # supaya labelnya konsisten dengan endpoint /predict yang sudah ada.
    p_bilstm, p_gru, p_cnn = _predict_single(text)
    return dict(merged_scores), p_bilstm, p_gru, p_cnn


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
    OPSI 3 — Prediksi + penjelasan XAI via SHAP wrapper black-box.
    Ensemble (bilstm+gru+cnn) di-treat sebagai satu fungsi predict tunggal;
    SHAP memperturbasi teks input (bukan gradien/index internal model) untuk
    menghitung kontribusi tiap kata. Lebih berat secara komputasi dibanding
    /predict biasa — cocok dipakai saat pengguna memang minta penjelasan,
    bukan dipanggil di setiap prediksi.
    """
    if app_state.get("explainer") is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability belum aktif — explainer gagal dibuat saat startup.",
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


@app.post("/debug/vocab-check", response_model=VocabCheckResponse, tags=["Debug"])
def vocab_check(request: VocabCheckRequest):
    """
    Cek apakah suatu kata dikenal (ada di vocabulary) tokenizer sequence
    dan/atau TF-IDF vectorizer, atau malah OOV (out-of-vocabulary).
    Berguna untuk investigasi kenapa suatu kata dapat skor SHAP yang
    janggal/tidak sesuai intuisi — kalau in_vocab=False, kata itu tidak
    pernah "dilihat" model dengan bentuk aslinya saat training.
    """
    tokenizer = app_state["tokenizer"]

    results = []
    for word in request.words:
        word_lower = word.lower()
        tok_idx = tokenizer.word_index.get(word_lower)
        tfidf_idx = app_state["tfidf"].vocabulary_.get(word_lower)

        results.append(VocabCheckResult(
            word=word,
            word_lower=word_lower,
            in_vocab=tok_idx is not None,
            tokenizer_index=tok_idx,
            tfidf_index=tfidf_idx,
        ))

    return VocabCheckResponse(results=results)