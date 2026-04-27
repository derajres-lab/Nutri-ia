import streamlit as st
import requests
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="NutriSnap AI", page_icon="🍎", layout="centered")

# Estética Profesional y Móvil
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: white; padding: 15px; border-radius: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    div.stButton > button { width: 100%; border-radius: 25px; height: 3em; background-color: #2E7D32; color: white; font-weight: bold; border: none; }
    </style>
    """, unsafe_allow_html=True)

# Encabezado
st.title("🍎 NutriSnap AI")
st.caption("Escáner nutricional gratuito, anónimo y privado")

# Inicializar historial
if 'dieta' not in st.session_state:
    st.session_state.dieta = []

# --- CARGAR MODELO YOLO ---
modelo = YOLO("yolov8n.pt")  # Modelo ligero, rápido y local


# --- MOTOR DE NUTRICIÓN ---
def buscar_nutrientes(nombre):
    """Busca en la base de datos gratuita de Open Food Facts"""
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={nombre}&search_simple=1&action=process&json=1"
    try:
        res = requests.get(url).json()
        if res['products']:
            prod = res['products'][0]
            nut = prod.get('nutriments', {})
            return {
                "nombre": prod.get('product_name', nombre),
                "kcal": nut.get('energy-kcal_100g', 0),
                "prot": nut.get('proteins_100g', 0),
                "carbs": nut.get('carbohydrates_100g', 0),
                "grasas": nut.get('fat_100g', 0)
            }
    except:
        return None
    return None


# --- INTERFAZ ---
tab1, tab2 = st.tabs(["📸 ESCANEAR", "📊 MI DÍA"])

# ---------------------------------------------------------
# 📸 TAB 1 — DETECCIÓN MÚLTIPLE CON YOLO
# ---------------------------------------------------------
with tab1:
    foto = st.camera_input("Enfoca tu comida")

    if foto:
        st.success("¡Imagen capturada!")

        # Convertir foto a matriz OpenCV
        img_bytes = foto.getvalue()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- DETECCIÓN MÚLTIPLE REAL ---
        resultados = modelo(img)

        detecciones = []
        for r in resultados:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detecciones.append({
                    "clase": modelo.names[cls],
                    "confianza": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        # Mostrar detecciones
        st.subheader("🍽️ Alimentos detectados:")

        if len(detecciones) == 0:
            st.warning("No se detectaron alimentos.")
        else:
            for d in detecciones:
                st.write(f"• **{d['clase']}** — {d['confianza']*100:.1f}%")

        # --- NUTRICIÓN ---
        st.subheader("📊 Información nutricional")

        for d in detecciones:
            datos = buscar_nutrientes(d["clase"])

            if datos:
                st.markdown(f"### {datos['nombre']}")
                c1, c2 = st.columns(2)
                c1.metric("Calorías", f"{datos['kcal']} kcal")
                c2.metric("Proteínas", f"{datos['prot']}g")

                c3, c4 = st.columns(2)
                c3.metric("Carbs", f"{datos['carbs']}g")
                c4.metric("Grasas", f"{datos['grasas']}g")

                if st.button(f"Añadir {datos['nombre']}"):
                    st.session_state.dieta.append(datos)
                    st.toast(f"{datos['nombre']} añadido")


# ---------------------------------------------------------
# 📊 TAB 2 — REGISTRO DIARIO
# ---------------------------------------------------------
with tab2:
    if not st.session_state.dieta:
        st.info("Aún no has registrado alimentos hoy.")
    else:
        total_cal = sum(d['kcal'] for d in st.session_state.dieta)
        st.header(f"Total: {total_cal:.0f} Kcal")

        for item in st.session_state.dieta:
            st.write(f"• **{item['nombre']}**: {item['kcal']} kcal")

        if st.button("🗑️ Borrar todo"):
            st.session_state.dieta = []
            st.rerun()

st.markdown("---")
st.caption("🔓 App 100% Gratuita | IA Local YOLOv8 | Datos: Open Food Facts")
