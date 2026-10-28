import pickle

import streamlit as st
import tensorflow as tf

import api


st.set_page_config(
    page_title="CekCok - Deteksi Berita Hoaks",
    page_icon="🔎",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_inference_resources() -> dict:
    """Muat model sekali untuk seluruh sesi Streamlit."""
    api._check_files()

    models = {
        "bilstm": tf.keras.models.load_model(
            api.MODEL_BILSTM,
            custom_objects=api.CUSTOM_OBJECTS,
            compile=False,
        ),
        "gru": tf.keras.models.load_model(
            api.MODEL_GRU,
            custom_objects=api.CUSTOM_OBJECTS,
            compile=False,
        ),
        "cnn": tf.keras.models.load_model(
            api.MODEL_CNN,
            custom_objects=api.CUSTOM_OBJECTS,
            compile=False,
        ),
    }

    vocab_sizes = {
        name: api._get_embedding_vocab_size(model)
        for name, model in models.items()
    }

    with open(api.TOKENIZER_BILSTM_PATH, "rb") as file:
        tokenizer_bilstm = pickle.load(file)
    with open(api.TOKENIZER_SHARED_PATH, "rb") as file:
        tokenizer_shared = pickle.load(file)
    with open(api.TFIDF_PATH, "rb") as file:
        tfidf = pickle.load(file)

    tokenizers = {
        "bilstm": tokenizer_bilstm,
        "gru": tokenizer_shared,
        "cnn": tokenizer_shared,
    }

    for name, tokenizer in tokenizers.items():
        if tokenizer.num_words != vocab_sizes[name]:
            raise RuntimeError(
                f"Tokenizer {name} tidak cocok dengan model: "
                f"num_words={tokenizer.num_words}, "
                f"embedding input_dim={vocab_sizes[name]}."
            )

    return {
        **models,
        "tokenizers": tokenizers,
        "tfidf": tfidf,
        "vocab_sizes": vocab_sizes,
    }


def activate_resources(resources: dict) -> None:
    """Hubungkan resource cache Streamlit ke helper inferensi di api.py."""
    api.app_state.update(
        {
            "bilstm": resources["bilstm"],
            "gru": resources["gru"],
            "cnn": resources["cnn"],
            "tokenizers": resources["tokenizers"],
            "tfidf": resources["tfidf"],
        }
    )
    api.VOCAB_SIZE.update(resources["vocab_sizes"])


def predict_text(text: str) -> api.PredictResponse:
    p_bilstm, p_gru, p_cnn = api._predict_single(text)
    return api._build_response(text, p_bilstm, p_gru, p_cnn)


def score_percent(score: float) -> str:
    return f"{score * 100:.2f}%"


st.title("🔎 CekCok")
st.caption("Deteksi berita hoaks dengan ensemble BiLSTM, GRU, dan CNN-BiLSTM.")

try:
    with st.spinner("Menyiapkan model deteksi..."):
        resources = load_inference_resources()
        activate_resources(resources)
except Exception as error:
    st.error("Model tidak dapat dimuat.")
    st.exception(error)
    st.stop()


single_tab, batch_tab = st.tabs(["Satu berita", "Beberapa berita"])

with single_tab:
    with st.form("single_prediction_form"):
        text = st.text_area(
            "Teks berita",
            height=220,
            placeholder="Tempelkan isi atau cuplikan berita di sini...",
        )
        submitted = st.form_submit_button(
            "Periksa berita",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        clean_text = text.strip()
        if not clean_text:
            st.warning("Masukkan teks berita terlebih dahulu.")
        else:
            try:
                with st.spinner("Menganalisis berita..."):
                    result = predict_text(clean_text)

                if result.is_fake:
                    st.error(f"Hasil: {result.label}")
                else:
                    st.success(f"Hasil: {result.label}")

                label_col, confidence_col = st.columns(2)
                label_col.metric("Klasifikasi", result.label)
                confidence_col.metric(
                    "Keyakinan hasil",
                    score_percent(result.confidence),
                )

                st.subheader("Skor tiap model")
                st.caption(
                    "Semakin mendekati 100%, model semakin condong menilai teks sebagai HOAKS."
                )
                scores = {
                    "BiLSTM": result.model_scores.bilstm,
                    "GRU": result.model_scores.gru,
                    "CNN-BiLSTM": result.model_scores.cnn_bilstm,
                }
                for model_name, score in scores.items():
                    st.write(f"**{model_name}:** {score_percent(score)}")
                    st.progress(max(0.0, min(1.0, score)))

                average_score = sum(scores.values()) / len(scores)
                st.caption(
                    f"Rata-rata skor HOAKS: {score_percent(average_score)} · "
                    f"Ambang klasifikasi: {score_percent(api.THRESHOLD)}"
                )
            except Exception as error:
                st.error("Prediksi gagal diproses.")
                st.exception(error)

with batch_tab:
    st.caption("Masukkan satu berita per baris, maksimal 50 berita.")
    with st.form("batch_prediction_form"):
        batch_text = st.text_area(
            "Daftar berita",
            height=220,
            placeholder="Berita pertama\nBerita kedua\nBerita ketiga",
        )
        batch_submitted = st.form_submit_button(
            "Periksa semua",
            type="primary",
            use_container_width=True,
        )

    if batch_submitted:
        texts = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not texts:
            st.warning("Masukkan setidaknya satu teks berita.")
        elif len(texts) > 50:
            st.warning("Maksimal 50 berita dalam satu pemeriksaan.")
        else:
            try:
                rows = []
                progress = st.progress(0.0, text="Memulai analisis...")
                for index, item in enumerate(texts, start=1):
                    result = predict_text(item)
                    rows.append(
                        {
                            "No.": index,
                            "Teks": item,
                            "Hasil": result.label,
                            "Keyakinan": score_percent(result.confidence),
                            "BiLSTM": score_percent(result.model_scores.bilstm),
                            "GRU": score_percent(result.model_scores.gru),
                            "CNN-BiLSTM": score_percent(result.model_scores.cnn_bilstm),
                        }
                    )
                    progress.progress(
                        index / len(texts),
                        text=f"Menganalisis {index} dari {len(texts)} berita...",
                    )

                progress.empty()
                st.dataframe(rows, hide_index=True, use_container_width=True)
            except Exception as error:
                st.error("Prediksi batch gagal diproses.")
                st.exception(error)

st.divider()
st.caption(
    "Hasil model merupakan alat bantu analisis dan sebaiknya tetap diverifikasi "
    "dengan sumber berita tepercaya."
)
