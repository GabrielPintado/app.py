import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import zscore

# --- 1. CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="DataMaster Pro v3", layout="wide", initial_sidebar_state="expanded")

# Estilos para asegurar el orden y marcos visuales
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
    .next-btn { border-top: 1px solid #ddd; padding-top: 20px; margin-top: 30px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DEL ESTADO DE SESIÓN ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'seccion_activa' not in st.session_state:
    st.session_state.seccion_activa = "📥 Etapa 0: Carga"

# Función para cambiar de etapa
def cambiar_etapa(nueva_etapa):
    st.session_state.seccion_activa = nueva_etapa
    st.rerun()

# --- 3. BARRA LATERAL ---
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
    # Sincronizamos el radio button con el estado de sesión
    seccion = st.radio("Ir a:", 
                      ["📥 Etapa 0: Carga", 
                       "📊 Etapa 1: Exploración (EDA)", 
                       "🧹 Etapa 2: Limpieza", 
                       "📈 Etapa 4: Visualización"],
                      key="navegacion_radio",
                      index=["📥 Etapa 0: Carga", 
                             "📊 Etapa 1: Exploración (EDA)", 
                             "🧹 Etapa 2: Limpieza", 
                             "📈 Etapa 4: Visualización"].index(st.session_state.seccion_activa))
    
    # Actualizar estado si el usuario cambia el radio manualmente
    if seccion != st.session_state.seccion_activa:
        st.session_state.seccion_activa = seccion

    if st.button("♻️ Reiniciar Dataset"):
        st.session_state.df = st.session_state.df_original.copy()
        st.rerun()

# --- 4. LÓGICA DE LAS ETAPAS ---

if st.session_state.df is None:
    st.header("Bienvenido a DataMaster Pro")
    st.info("Por favor, sube un archivo CSV en el panel de la izquierda para comenzar.")
    st.stop()

df = st.session_state.df

# --- ETAPA 0: CARGA ---
if st.session_state.seccion_activa == "📥 Etapa 0: Carga":
    st.header("Exploración Inicial")
    st.write("Datos cargados actualmente:")
    st.dataframe(df.head(15), use_container_width=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Filas", df.shape[0])
    col_m2.metric("Total Columnas", df.shape[1])
    col_m3.metric("Datos Nulos", df.isna().sum().sum())
    
    st.markdown('<div class="next-btn"></div>', unsafe_allow_html=True)
    if st.button("Siguiente: Ir a Exploración (EDA) ➔"):
        cambiar_etapa("📊 Etapa 1: Exploración (EDA)")

# --- ETAPA 1: EDA ---
elif st.session_state.seccion_activa == "📊 Etapa 1: Exploración (EDA)":
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
    
    st.markdown('<div class="next-btn"></div>', unsafe_allow_html=True)
    if st.button("Siguiente: Ir a Limpieza ➔"):
        cambiar_etapa("🧹 Etapa 2: Limpieza")

# --- ETAPA 2: LIMPIEZA ---
elif st.session_state.seccion_activa == "🧹 Etapa 2: Limpieza":
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

    st.markdown('<div class="next-btn"></div>', unsafe_allow_html=True)
    if st.button("Siguiente: Ir a Visualización ➔"):
        cambiar_etapa("📈 Etapa 4: Visualización")

# --- ETAPA 4: VISUALIZACIÓN ---
elif st.session_state.seccion_activa == "📈 Etapa 4: Visualización":
    st.header("Visualización de Datos Dinámica")
    
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()
    
    tipo_g = st.selectbox("Tipo de Gráfico", ["Correlación", "Distribución (Violín/Box)", "Barras de Error", "Dispersión"])
    
    st.divider()
    
    if tipo_g == "Correlación":
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
            st.plotly_chart(px.violin(df, y=col_y, x=col_x, box=True, points="all", title=f"Violin Plot: {col_y}"), use_container_width=True)
        with c2:
            st.plotly_chart(px.box(df, y=col_y, x=col_x, notched=True, title=f"Box Plot: {col_y}"), use_container_width=True)

    elif tipo_g == "Barras de Error":
        if len(cols_cat) > 0:
            cat = st.selectbox("Variable Categórica:", cols_cat)
            num = st.selectbox("Variable Numérica:", cols_num)
            df_err = df.groupby(cat)[num].agg(['mean', 'std']).reset_index()
            st.plotly_chart(px.bar(df_err, x=cat, y='mean', error_y='std', title=f"Promedio de {num}"), use_container_width=True)
        else:
            st.error("No hay columnas categóricas.")

    elif tipo_g == "Dispersión":
        x_axis = st.selectbox("Eje X:", cols_num)
        y_axis = st.selectbox("Eje Y:", cols_num, index=1 if len(cols_num)>1 else 0)
        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title="Correlación Visual"), use_container_width=True)

    st.markdown('<div class="next-btn"></div>', unsafe_allow_html=True)
    if st.button("Finalizar y Reiniciar Flujo"):
        cambiar_etapa("📥 Etapa 0: Carga")
