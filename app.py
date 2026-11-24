
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Reciclaje Computación en la Nube",
    layout="centered"
)

st.title("Modelo predictivo Reciclaje clase de IA-UTH -2026-Luis Diaz")
st.write("Suba o tome una fotografía para clasificar con el modelo MobileNetV2 preentrenado.")

IMG_SIZE = (224, 224)

MODEL_DIR = Path("modelo_reciclaje_mobilenet")
CLASS_PATH = MODEL_DIR / "class_names.json"
MODEL_PATHS = [
    MODEL_DIR / "waste_mobilenet.keras",
    MODEL_DIR / "waste_mobilenet.h5"
]

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}


@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)

    st.error(
        "No se encontró el modelo. Coloque la carpeta "
        "modelo_reciclaje_mobilenet junto a app.py."
    )
    st.stop()


@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]

    top3 = np.argsort(preds)[-3:][::-1]

    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top3
    ]


# Cargar modelo y clases
modelo = cargar_modelo()
clases = cargar_clases()


# Selección del método para proporcionar la imagen
st.subheader("Seleccionar imagen")

opcion = st.radio(
    "¿Cómo desea proporcionar la imagen?",
    ["Tomar fotografía", "Seleccionar de galería"],
    horizontal=False
)

archivo = None

if opcion == "Tomar fotografía":

    archivo = st.camera_input(
        "Abra la cámara y tome una fotografía"
    )

else:

    archivo = st.file_uploader(
        "Seleccione una imagen de su dispositivo",
        type=["jpg", "jpeg", "png", "webp"]
    )


# Procesar y clasificar imagen
if archivo is not None:

    try:
        imagen = Image.open(archivo).convert("RGB")

        st.image(
            imagen,
            caption="Imagen analizada",
            use_container_width=True
        )

        resultados = predecir(imagen)

        st.subheader("Resultado")

        st.success(
            f"Predicción principal: {resultados[0][0]} "
            f"({resultados[0][1]:.2f}%)"
        )

        st.write("Top 3 probabilidades:")

        for clase, prob in resultados:
            st.write(f"{clase}: {prob:.2f}%")

    except Exception as e:
        st.error(f"No se pudo procesar la imagen: {e}")

else:

    st.info(
        "Tome una fotografía o seleccione una imagen "
        "de la galería para iniciar la clasificación."
    )
