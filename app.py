import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import zscore

# 1. Configuración de Estética y Layout
st.set_page_config(page_title="DataMaster Pro v2", layout="wide")

# CSS para mejorar el orden y crear "Tarjetas" visuales
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stRadio > div { flex-direction: row; gap: 15px; }
    .st-emotion-cache-12w0qpk { border: 1px solid #ddd; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    h1, h2 { color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

# --- ESTADO DE LA APLICACIÓN ---
if 'df' not in st.session_state:
    st.session_state.df = None

# --- SIDEBAR PERSISTENTE ---
with st.sidebar:
    st.title("📂 Gestión de Datos")
    archivo = st.file_uploader("Cargar CSV", type=["csv"])
    if archivo:
        if st.session_state.df is None or st.button("🔄 Reiniciar Dataset"):
            st.session_state.df = pd.read_csv(archivo)
            st.success("¡Datos cargados!")
    
    st.divider()
    st.markdown("### Navegación")
    etapa = st.radio("Selecciona Fase:", 
                    ["0. Carga", "1. Exploración", "2. Limpieza", "3. Visualización"])

# --- LÓGICA DE NAVEGACIÓN ---

# Si no hay datos, forzar permanencia en etapa 0
if st.session_state.df is None:
    st.header("📥 Etapa 0: Ingesta de Datos")
    st.info("Por favor, sube un archivo CSV en el panel lateral para desbloquear las herramientas.")
    st.stop()

df = st.session_state.df

# ETAPA 1: EXPLORACIÓN
if etapa == "1. Exploración":
    st.header("📊 Etapa 1: Análisis Exploratorio (EDA)")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        with st.expander("👀 Vista Previa (Primeros 5)", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
    with col_b:
        with st.expander("📝 Resumen Estadístico", expanded=True):
            st.dataframe(df.describe().T, use_container_width=True)

    st.divider()
    st.subheader("🔍 Buscador de Correlación Rápida")
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    if len(cols_num) >= 2:
        c1, c2 = st.columns(2)
        v1 = c1.selectbox("Variable 1", cols_num, index=0)
        v2 = c2.selectbox("Variable 2", cols_num, index=1 if len(cols_num)>1 else 0)
        corr_val = df[v1].corr(df[v2])
        st.metric(f"Correlación entre {v1} y {v2}", f"{corr_val:.2f}")
    else:
        st.warning("Se necesitan al menos 2 columnas numéricas.")

# ETAPA 2: LIMPIEZA
elif etapa == "2. Limpieza":
    st.header("🧹 Etapa 2: Curación y Estandarización")
    
    tab_n, tab_s = st.tabs(["Tratamiento de Datos", "Estandarización Z-Score"])
    
    with tab_n:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Eliminar Filas con Nulos"):
                st.session_state.df = df.dropna()
                st.success("Nulos eliminados")
                st.rerun()
        with c2:
            if st.button("👯 Eliminar Duplicados"):
                st.session_state.df = df.drop_duplicates()
                st.success("Duplicados eliminados")
                st.rerun()

    with tab_s:
        cols_num = df.select_dtypes(include=np.number).columns.tolist()
        to_scale = st.multiselect("Selecciona columnas para estandarizar:", cols_num)
        if st.button("⚙️ Aplicar Estandarización"):
            for c in to_scale:
                st.session_state.df[f"{c}_std"] = zscore(df[c], ddof=1)
            st.success("Nuevas columnas *_std añadidas.")
            st.rerun()

# ETAPA 3: VISUALIZACIÓN
elif etapa == "3. Visualización":
    st.header("🎨 Etapa 3: Laboratorio Visual")
    
    # Selector de gráfico en el área principal para mayor visibilidad
    tipo = st.selectbox("Elige el diagrama:", 
                       ["Matriz de Correlación", "Violín Interactivo", "Cajas (Boxplot)", "Barras de Error"])
    
    st.markdown("---")
    
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()

    if tipo == "Matriz de Correlación":
        st.subheader("🔥 Mapa Térmico de Correlación")
        fig = px.imshow(df[cols_num].corr(), text_auto=".2f", color_continuous_scale='RdBu_r', range_color=[-1,1])
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Violín Interactivo":
        col_y = st.selectbox("Eje Y (Numérico):", cols_num)
        col_x = st.selectbox("Eje X (Categoría):", [None] + cols_cat)
        fig = px.violin(df, y=col_y, x=col_x, box=True, points="all", color=col_x if col_x else None,
                       title=f"Distribución de {col_y} con Límites de Datos")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Cajas (Boxplot)":
        col_y = st.selectbox("Eje Y:", cols_num)
        fig = px.box(df, y=col_y, notched=True, title=f"Análisis de Cuartiles y Outliers: {col_y}")
        st.plotly_chart(fig, use_container_width=True)

    elif tipo == "Barras de Error":
        if len(cols_cat) > 0:
            x = st.selectbox("Categoría:", cols_cat)
            y = st.selectbox("Valor:", cols_num)
            df_stats = df.groupby(x)[y].agg(['mean', 'std']).reset_index()
            fig = px.bar(df_stats, x=x, y='mean', error_y='std', title=f"Promedio y Dispersión de {y}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No hay columnas categóricas para este gráfico.")
