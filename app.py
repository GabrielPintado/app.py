import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import zscore

# --- 1. CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="DataMaster Pro v3", layout="wide", initial_sidebar_state="expanded")

# Estilos para marcos, métricas y diseño limpio
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric { background-color: #ffffff; border: 1px solid #ddd; padding: 15px; border-radius: 10px; }
    div[data-testid="stVerticalBlock"] > div:has(div.stPlot) {
        border: 1px solid #e6e9ef;
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DEL ESTADO DE SESIÓN ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None

# Lista maestra de etapas
ETAPAS = [
    "📥 Etapa 0: Carga", 
    "📊 Etapa 1: Exploración (EDA)", 
    "🧹 Etapa 2: Limpieza", 
    "📈 Etapa 4: Visualización"
]

# Inicializar controlador de navegación si no existe
if 'nav_key' not in st.session_state:
    st.session_state.nav_key = ETAPAS[0]

# Función para saltar de etapa
def saltar_a(indice):
    st.session_state.nav_key = ETAPAS[indice]

# --- 3. BARRA LATERAL (NAVEGACIÓN Y CARGA) ---
with st.sidebar:
    st.title("🛡️ Data Engine")
    st.subheader("Carga de Archivo")
    archivo = st.file_uploader("Sube un CSV", type=["csv"])
    
    if archivo is not None:
        if st.session_state.df_original is None:
            df_init = pd.read_csv(archivo)
            st.session_state.df = df_init
            st.session_state.df_original = df_init.copy()
            st.success("¡Archivo cargado!")
    
    st.divider()
    st.subheader("Navegación")
    # El Radio Button está vinculado a 'nav_key'
    seccion = st.radio("Ir a:", ETAPAS, key="nav_key")
    
    if st.button("♻️ Reiniciar Dataset"):
        st.session_state.df = st.session_state.df_original.copy()
        st.rerun()

# --- 4. LÓGICA DE LAS ETAPAS ---

# Verificación de datos global
if st.session_state.df is None:
    st.header("Bienvenido a DataMaster Pro")
    st.info("Por favor, sube un archivo CSV en el panel de la izquierda para comenzar.")
    st.stop()

df = st.session_state.df

# --- ETAPA 0: CARGA ---
if seccion == ETAPAS[0]:
    st.header("Exploración Inicial")
    st.write("Datos cargados actualmente:")
    st.dataframe(df.head(15), use_container_width=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Filas", df.shape[0])
    col_m2.metric("Total Columnas", df.shape[1])
    col_m3.metric("Datos Nulos", df.isna().sum().sum())
    
    st.divider()
    # Botón de navegación al final
    st.button("Siguiente: Ir a Exploración (EDA) ➔", on_click=saltar_a, args=(1,))

# --- ETAPA 1: EDA ---
elif seccion == ETAPAS[1]:
    st.header("Análisis Exploratorio de Datos")
    
    with st.container():
        st.subheader("Información de las Variables")
        info_df = pd.DataFrame({
            "Tipo": df.dtypes.astype(str),
            "No Nulos": df.count(),
            "Nulos": df.isnull().sum(),
            "Unicos": df.nunique()
        })
        st.table(info_df)
    
    st.divider()
    st.subheader("Estadísticas de Tendencia Central")
    st.dataframe(df.describe().T, use_container_width=True)
    
    st.divider()
    st.button("Siguiente: Ir a Limpieza ➔", on_click=saltar_a, args=(2,))

# --- ETAPA 2: LIMPIEZA ---
elif seccion == ETAPAS[2]:
    st.header("Procesamiento y Curación")
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Limpieza Básica")
        if st.button("Eliminar Filas con Nulos"):
            st.session_state.df = df.dropna()
            st.success("Nulos eliminados")
            st.rerun()
            
        if st.button("Eliminar Duplicados"):
            st.session_state.df = df.drop_duplicates()
            st.success("Duplicados eliminados")
            st.rerun()

    with col_r:
        st.subheader("Estandarización (Z-Score)")
        cols_num = df.select_dtypes(include=np.number).columns.tolist()
        target = st.multiselect("Columnas a normalizar:", cols_num)
        if st.button("Aplicar Estandarización"):
            for c in target:
                st.session_state.df[f"{c}_std"] = zscore(df[c], ddof=1)
            st.success("Columnas creadas con sufijo _std")
            st.rerun()
            
    st.divider()
    st.button("Siguiente: Ir a Visualización ➔", on_click=saltar_a, args=(3,))

# --- ETAPA 4: VISUALIZACIÓN ---
elif seccion == ETAPAS[3]:
    st.header("Visualización de Datos Dinámica")
    
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()
    
    tipo_g = st.selectbox("Tipo de Gráfico", ["Correlación", "Distribución (Violín/Box)", "Barras de Error", "Dispersión"])
    
    st.divider()
    
    if tipo_g == "Correlación":
        st.subheader("Matriz de Correlación de Pearson")
        if len(cols_num) > 1:
            fig = px.imshow(df[cols_num].corr(), text_auto=".2f", color_continuous_scale='RdBu_r', range_color=[-1,1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Se necesitan al menos 2 columnas numéricas.")

    elif tipo_g == "Distribución (Violín/Box)":
        col_y = st.selectbox("Selecciona Variable Numérica:", cols_num)
        col_x = st.selectbox("Agrupar por (Opcional):", [None] + cols_cat)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_v = px.violin(df, y=col_y, x=col_x, box=True, points="all", title=f"Violin Plot: {col_y}")
            st.plotly_chart(fig_v, use_container_width=True)
        with c2:
            fig_b = px.box(df, y=col_y, x=col_x, notched=True, title=f"Box Plot: {col_y}")
            st.plotly_chart(fig_b, use_container_width=True)

    elif tipo_g == "Barras de Error":
        if len(cols_cat) > 0:
            cat = st.selectbox("Variable Categórica:", cols_cat)
            num = st.selectbox("Variable Numérica:", cols_num)
            df_err = df.groupby(cat)[num].agg(['mean', 'std']).reset_index()
            fig = px.bar(df_err, x=cat, y='mean', error_y='std', title=f"Promedio de {num} con Desviación Estándar")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No hay columnas categóricas para agrupar.")

    elif tipo_g == "Dispersión":
        x_axis = st.selectbox("Eje X:", cols_num)
        y_axis = st.selectbox("Eje Y:", cols_num, index=1 if len(cols_num)>1 else 0)
        fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=f"Correlación Visual: {x_axis} vs {y_axis}")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.button("Reiniciar al Inicio (Carga) ⟲", on_click=saltar_a, args=(0,))
