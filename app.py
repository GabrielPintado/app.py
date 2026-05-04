import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. Configuración de la página y Estilos CSS para limpieza visual
st.set_page_config(page_title="DataMaster Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- NAVEGACIÓN PERSISTENTE (SIDEBAR) ---
st.sidebar.title("🎮 Panel de Control")
st.sidebar.markdown("---")

# Inicializar estados
if 'df' not in st.session_state:
    st.session_state.df = None
if 'page' not in st.session_state:
    st.session_state.page = "Etapa 0: Carga"

# Función para cambiar de página desde cualquier lugar
def nav_to(page):
    st.session_state.page = page

# Menú de navegación en el sidebar (Siempre visible)
opciones = ["Etapa 0: Carga", "Etapa 1: EDA Profundo", "Etapa 2: Limpieza", "Etapa 4: Visualización"]
page = st.sidebar.radio("Ir a la etapa:", opciones, index=opciones.index(st.session_state.page))
st.session_state.page = page

# --- LÓGICA DE ETAPAS ---

# ETAPA 0: CARGA
if st.session_state.page == "Etapa 0: Carga":
    st.header("📥 Etapa 0: Ingesta de Datos")
    archivo = st.file_uploader("Sube tu archivo CSV para comenzar", type=["csv"])
    
    if archivo:
        st.session_state.df = pd.read_csv(archivo)
        st.success("✅ ¡Datos cargados exitosamente!")
        st.subheader("Vista Previa (Primeros 10 registros)")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        if st.button("Siguiente: Ir a EDA ➔"):
            nav_to("Etapa 1: EDA Profundo")
            st.rerun()

# VALIDACIÓN DE DATOS ANTES DE SEGUIR
if st.session_state.df is None and st.session_state.page != "Etapa 0: Carga":
    st.warning("⚠️ Primero debes cargar un archivo en la Etapa 0.")
    st.stop()

# ETAPA 1: EDA PROFUNDO
elif st.session_state.page == "Etapa 1: EDA Profundo":
    st.header("📊 Etapa 1: Análisis Exploratorio Profundo")
    df = st.session_state.df
    
    with st.container():
        st.subheader("📋 Resumen Estadístico")
        st.dataframe(df.describe().T, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧩 Tipos de Datos")
            st.write(df.dtypes)
        with col2:
            st.subheader("❓ Análisis de Nulos")
            st.write(df.isnull().sum())

    st.divider()
    st.subheader("📈 Análisis de Distribución (Momentos)")
    cols_num = df.select_dtypes(include=np.number).columns
    if not cols_num.empty:
        selected_col = st.selectbox("Elige una columna numérica:", cols_num)
        c1, c2, c3 = st.columns(3)
        c1.metric("Asimetría (Skewness)", round(df[selected_col].skew(), 2))
        c2.metric("Curtosis", round(df[selected_col].kurt(), 2))
        c3.metric("Media", round(df[selected_col].mean(), 2))
        
        fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribución de {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

# ETAPA 2: LIMPIEZA
elif st.session_state.page == "Etapa 2: Limpieza":
    st.header("🧹 Etapa 2: Limpieza y Curación")
    df = st.session_state.df.copy()
    
    st.info("Nota: Las acciones aplicadas aquí actualizarán el dataset global.")
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.subheader("1. Gestión de Nulos")
        metodo = st.selectbox("Estrategia:", ["Ninguna", "Eliminar Filas con Nulos", "Llenar con Media", "Llenar con Mediana"])
        if st.button("Aplicar Tratamiento de Nulos"):
            if metodo == "Eliminar Filas con Nulos": df = df.dropna()
            elif metodo == "Llenar con Media": df = df.fillna(df.mean(numeric_only=True))
            elif metodo == "Llenar con Mediana": df = df.fillna(df.median(numeric_only=True))
            st.session_state.df = df
            st.success("¡Nulos tratados!")

    with col_l2:
        st.subheader("2. Duplicados")
        if st.button("Remover Registros Duplicados"):
            antes = len(df)
            df = df.drop_duplicates()
            st.session_state.df = df
            st.write(f"Filas eliminadas: {antes - len(df)}")

    st.divider()
    st.subheader("3. Tratamiento de Outliers (IQR)")
    cols_num = df.select_dtypes(include=np.number).columns
    out_col = st.selectbox("Selecciona columna para limpiar outliers:", cols_num)
    if st.button("Limpiar Outliers"):
        Q1 = df[out_col].quantile(0.25)
        Q3 = df[out_col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[out_col] < (Q1 - 1.5 * IQR)) | (df[out_col] > (Q3 + 1.5 * IQR)))]
        st.session_state.df = df
        st.rerun()

# ETAPA 4: VISUALIZACIÓN
elif st.session_state.page == "Etapa 4: Visualización":
    st.header("🎨 Etapa 4: Visualización Interactiva Avanzada")
    df = st.session_state.df
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()

    viz_choice = st.selectbox("Selecciona el tipo de diagrama:", 
                               ["Mapa de Correlación", "Diagrama de Violín", "Boxplot", "Gráfico de Barras con Error", "Dispersión con Error"])

    if viz_choice == "Mapa de Correlación":
        st.subheader("Relación entre Variables Numéricas")
        fig = px.imshow(df[cols_num].corr(), text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    elif viz_choice == "Diagrama de Violín":
        y_col = st.selectbox("Variable numérica (Eje Y):", cols_num)
        x_col = st.selectbox("Agrupar por (Categoría - Opcional):", [None] + cols_cat)
        fig = px.violin(df, y=y_col, x=x_col, box=True, points="all", title=f"Violin Plot de {y_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_choice == "Boxplot":
        y_col = st.selectbox("Variable numérica:", cols_num)
        x_col = st.selectbox("Categoría:", [None] + cols_cat)
        fig = px.box(df, y=y_col, x=x_col, notched=True, title=f"Análisis de Cuartiles: {y_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_choice == "Gráfico de Barras con Error":
        y_col = st.selectbox("Valor (Y):", cols_num)
        x_col = st.selectbox("Categoría (X):", cols_cat)
        # Cálculo simple de error (desviación estándar) para el ejemplo
        df_stats = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        fig = px.bar(df_stats, x=x_col, y='mean', error_y='std', title=f"Promedio de {y_col} con Desviación Estándar")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_choice == "Dispersión con Error":
        x_col = st.selectbox("Eje X:", cols_num)
        y_col = st.selectbox("Eje Y:", cols_num)
        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"Regresión Lineal: {x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)
