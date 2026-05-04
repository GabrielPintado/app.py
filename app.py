import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuración de página
st.set_page_config(page_title="DataExplorer Pro", layout="wide", initial_sidebar_state="expanded")

# Título Principal
st.title("🔬 DataExplorer Pro: Suite de Análisis Modular")

# Inicialización de estado para persistencia de datos
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar - Configuración Global
st.sidebar.header("🛠️ Configuración de Datos")
archivo = st.sidebar.file_uploader("Cargar dataset (CSV)", type=["csv"])

if archivo:
    if st.session_state.df is None or st.sidebar.button("Recargar Archivo"):
        st.session_state.df = pd.read_csv(archivo)
        st.sidebar.success("¡Archivo cargado!")

# Lógica de Pestañas
if st.session_state.df is not None:
    tab0, tab1, tab2, tab4 = st.tabs([
        "📥 Etapa 0: Carga", 
        "📊 Etapa 1: EDA Profundo", 
        "🧹 Etapa 2: Limpieza Avanzada", 
        "📈 Etapa 4: Visualización"
    ])

    # --- ETAPA 0: CARGA ---
    with tab0:
        st.header("Inspección Inicial de Datos")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Filas", st.session_state.df.shape[0])
        col2.metric("Total Columnas", st.session_state.df.shape[1])
        col3.metric("Memoria (MB)", round(st.session_state.df.memory_usage().sum() / 1024**2, 2))

    # --- ETAPA 1: EDA ---
    with tab1:
        st.header("Análisis Exploratorio de Datos (EDA)")
        col_eda1, col_eda2 = st.columns([1, 2])
        
        with col_eda1:
            st.subheader("Tipos de Datos y Nulos")
            info_df = pd.DataFrame({
                "Tipo": st.session_state.df.dtypes,
                "Nulos": st.session_state.df.isnull().sum(),
                "% Nulos": (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2)
            })
            st.table(info_df)
            
        with col_eda2:
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        st.divider()
        st.subheader("Distribución de Variables Numéricas")
        cols_num = st.session_state.df.select_dtypes(include=np.number).columns
        if not cols_num.empty:
            target_col = st.selectbox("Selecciona columna para ver distribución:", cols_num)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(st.session_state.df[target_col], kde=True, color="skyblue", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No hay columnas numéricas para graficar distribuciones.")

    # --- ETAPA 2: LIMPIEZA ---
    with tab2:
        st.header("Motor de Limpieza y Curación")
        df_clean = st.session_state.df.copy()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tratamiento de Nulos")
            metodo_nulos = st.selectbox("Estrategia para Nulos:", ["Nada", "Eliminar Filas", "Llenar con Media", "Llenar con Mediana"])
            if st.button("Aplicar Estrategia Nulos"):
                if metodo_nulos == "Eliminar Filas":
                    df_clean.dropna(inplace=True)
                elif metodo_nulos == "Llenar con Media":
                    df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)
                elif metodo_nulos == "Llenar con Mediana":
                    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
                st.session_state.df = df_clean
                st.success("Limpieza aplicada.")

        with c2:
            st.subheader("Duplicados y Outliers")
            if st.button("Eliminar Duplicados"):
                df_clean.drop_duplicates(inplace=True)
                st.session_state.df = df_clean
                st.success("Duplicados eliminados.")
            
            outlier_col = st.selectbox("Remover Outliers (IQR) en:", cols_num)
            if st.button("Limpiar Outliers"):
                Q1 = df_clean[outlier_col].quantile(0.25)
                Q3 = df_clean[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean = df_clean[~((df_clean[outlier_col] < (Q1 - 1.5 * IQR)) | (df_clean[outlier_col] > (Q3 + 1.5 * IQR)))]
                st.session_state.df = df_clean
                st.rerun()

    # --- ETAPA 4: VISUALIZACIÓN ---
    with tab4:
        st.header("Laboratorio de Visualización")
        
        v_type = st.radio("Tipo de Gráfico:", ["Mapa de Correlación", "Dispersión (Scatter)", "Boxplot de Comparación"], horizontal=True)
        
        if v_type == "Mapa de Correlación":
            cols_corr = st.multiselect("Columnas para Correlación:", cols_num, default=list(cols_num))
            if len(cols_corr) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(st.session_state.df[cols_corr].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Selecciona al menos 2 columnas.")
                
        elif v_type == "Dispersión (Scatter)":
            c_x = st.selectbox("Eje X:", cols_num)
            c_y = st.selectbox("Eje Y:", cols_num)
            c_hue = st.selectbox("Color por (Categoría):", [None] + list(st.session_state.df.columns))
            fig, ax = plt.subplots()
            sns.scatterplot(data=st.session_state.df, x=c_x, y=c_y, hue=c_hue, ax=ax)
            st.pyplot(fig)

        elif v_type == "Boxplot de Comparación":
            c_num = st.selectbox("Variable Numérica:", cols_num)
            c_cat = st.selectbox("Agrupar por:", st.session_state.df.select_dtypes(include='object').columns)
            fig, ax = plt.subplots()
            sns.boxplot(data=st.session_state.df, x=c_cat, y=c_num, ax=ax)
            st.pyplot(fig)
else:
    st.info("👋 Bienvenid@. Por favor, carga un archivo CSV en el panel lateral para comenzar el análisis.")
