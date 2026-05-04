import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import zscore

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="DataMaster Pro v3", layout="wide")

# --- 2. GESTIÓN DEL ESTADO (Session State) ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None

# Lista de etapas para manejo de índices
ETAPAS = [
    "📥 Etapa 0: Carga", 
    "📊 Etapa 1: Exploración (EDA)", 
    "🧹 Etapa 2: Limpieza", 
    "📈 Etapa 4: Visualización"
]

# Inicializar la llave del radio si no existe
if 'nav_key' not in st.session_state:
    st.session_state.nav_key = ETAPAS[0]

# Función para mover a la siguiente etapa
def mover_a_etapa(indice):
    st.session_state.nav_key = ETAPAS[indice]

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.title("🛡️ Data Engine")
    archivo = st.file_uploader("Sube un CSV", type=["csv"])
    
    if archivo:
        if st.session_state.df_original is None:
            df_init = pd.read_csv(archivo)
            st.session_state.df = df_init
            st.session_state.df_original = df_init.copy()

    st.divider()
    # El radio button ahora usa 'nav_key' como su estado maestro
    seccion = st.radio("Navegación:", ETAPAS, key="nav_key")
    
    if st.button("♻️ Reiniciar Dataset"):
        st.session_state.df = st.session_state.df_original.copy()
        st.rerun()

# --- 4. LÓGICA DE LAS ETAPAS ---

if st.session_state.df is None:
    st.header("Bienvenido a DataMaster Pro")
    st.info("Sube un CSV en el panel izquierdo para comenzar.")
    st.stop()

df = st.session_state.df

# --- ETAPA 0: CARGA ---
if seccion == ETAPAS[0]:
    st.header("Exploración Inicial")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Nulos", df.isna().sum().sum())
    
    st.divider()
    # El botón ahora llama a la función para cambiar el índice del radio
    st.button("Siguiente: Ir a Exploración ➔", on_click=mover_a_etapa, args=(1,))

# --- ETAPA 1: EDA ---
elif seccion == ETAPAS[1]:
    st.header("Análisis Exploratorio")
    st.subheader("Estadísticas Generales")
    st.dataframe(df.describe().T, use_container_width=True)
    
    st.divider()
    st.button("Siguiente: Ir a Limpieza ➔", on_click=mover_a_etapa, args=(2,))

# --- ETAPA 2: LIMPIEZA ---
elif seccion == ETAPAS[2]:
    st.header("Limpieza y Procesamiento")
    
    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("Eliminar Filas con Nulos"):
            st.session_state.df = df.dropna()
            st.rerun()
    with col_r:
        cols_num = df.select_dtypes(include=np.number).columns.tolist()
        target = st.multiselect("Estandarizar:", cols_num)
        if st.button("Aplicar Z-Score"):
            for c in target:
                st.session_state.df[f"{c}_std"] = zscore(df[c], ddof=1)
            st.rerun()

    st.divider()
    st.button("Siguiente: Ir a Visualización ➔", on_click=mover_a_etapa, args=(3,))

# --- ETAPA 4: VISUALIZACIÓN ---
elif seccion == ETAPAS[3]:
    st.header("Visualización Interactiva")
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    
    tipo_g = st.selectbox("Gráfico:", ["Correlación", "Distribución", "Dispersión"])
    
    if tipo_g == "Correlación":
        fig = px.imshow(df[cols_num].corr(), text_auto=".2f", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    elif tipo_g == "Distribución":
        col = st.selectbox("Variable:", cols_num)
        st.plotly_chart(px.violin(df, y=col, box=True, points="all"), use_container_width=True)

    elif tipo_g == "Dispersión":
        x = st.selectbox("Eje X:", cols_num)
        y = st.selectbox("Eje Y:", cols_num, index=1 if len(cols_num)>1 else 0)
        st.plotly_chart(px.scatter(df, x=x, y=y, trendline="ols"), use_container_width=True)

    st.divider()
    st.button("Reiniciar al Inicio ⟲", on_click=mover_a_etapa, args=(0,))
