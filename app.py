import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.stats import zscore

# Configuración y Estilos
st.set_page_config(page_title="DataMaster Pro", layout="wide")

# --- NAVEGACIÓN PERSISTENTE ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'page' not in st.session_state:
    st.session_state.page = "Etapa 0: Carga"

def nav_to(page):
    st.session_state.page = page

st.sidebar.title("🎮 Panel de Control")
opciones = ["Etapa 0: Carga", "Etapa 1: EDA Profundo", "Etapa 2: Limpieza", "Etapa 4: Visualización"]
page = st.sidebar.radio("Ir a la etapa:", opciones, index=opciones.index(st.session_state.page))
st.session_state.page = page

# --- LÓGICA DE ETAPAS ---

if st.session_state.page == "Etapa 0: Carga":
    st.header("📥 Etapa 0: Ingesta de Datos")
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo:
        st.session_state.df = pd.read_csv(archivo)
        st.success("✅ ¡Datos cargados!")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        if st.button("Siguiente: Ir a EDA ➔"): nav_to("Etapa 1: EDA Profundo"); st.rerun()

elif st.session_state.df is None:
    st.warning("⚠️ Carga un archivo en la Etapa 0.")
    st.stop()

elif st.session_state.page == "Etapa 1: EDA Profundo":
    st.header("📊 Etapa 1: EDA Profundo")
    df = st.session_state.df
    st.subheader("📋 Resumen Estadístico")
    st.dataframe(df.describe().T, use_container_width=True)
    
    cols_num = df.select_dtypes(include=np.number).columns
    if not cols_num.empty:
        selected_col = st.selectbox("Análisis de Distribución:", cols_num)
        fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribución de {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.page == "Etapa 2: Limpieza":
    st.header("🧹 Etapa 2: Limpieza y Estandarización")
    df = st.session_state.df.copy()
    
    # --- SUBSECCIÓN: LIMPIEZA BÁSICA ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Gestión de Nulos")
        metodo = st.selectbox("Estrategia:", ["Ninguna", "Eliminar Filas", "Llenar con Media"])
        if st.button("Tratar Nulos"):
            if metodo == "Eliminar Filas": df = df.dropna()
            elif metodo == "Llenar con Media": df = df.fillna(df.mean(numeric_only=True))
            st.session_state.df = df
            st.success("¡Nulos procesados!")

    with col2:
        st.subheader("2. Outliers (IQR)")
        cols_num = df.select_dtypes(include=np.number).columns
        out_col = st.selectbox("Columna para Outliers:", cols_num)
        if st.button("Limpiar Outliers"):
            q1, q3 = df[out_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[~((df[out_col] < (q1 - 1.5 * iqr)) | (df[out_col] > (q3 + 1.5 * iqr)))]
            st.session_state.df = df
            st.rerun()

    st.divider()

    # --- NUEVA SUBSECCIÓN: ESTANDARIZACIÓN ---
    st.subheader("3. Estandarización de Datos (Z-Score)")
    st.markdown("""
    La estandarización transforma los datos para que tengan **media = 0** y **desviación estándar = 1**. 
    Es útil para comparar variables con diferentes escalas (ej: Edad vs Salario).
    """)
    
    cols_to_scale = st.multiselect("Selecciona columnas para estandarizar:", cols_num)
    
    if st.button("Estandarizar seleccionadas"):
        if cols_to_scale:
            for col in cols_to_scale:
                # Aplicamos Z-score: (x - mean) / std
                df[f"{col}_std"] = zscore(df[col], ddof=1)
            
            st.session_state.df = df
            st.success(f"✅ Se han creado {len(cols_to_scale)} nuevas columnas estandarizadas (con sufijo _std).")
            st.dataframe(df[[c for c in df.columns if "_std" in c]].head(), use_container_width=True)
        else:
            st.warning("Selecciona al menos una columna.")

elif st.session_state.page == "Etapa 4: Visualización":
    st.header("🎨 Etapa 4: Visualización")
    df = st.session_state.df
    cols_num = df.select_dtypes(include=np.number).columns.tolist()
    cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()

    tipo = st.selectbox("Tipo de Gráfico:", ["Violín", "Boxplot", "Barras con Error", "Correlación"])

    if tipo == "Violín":
        y = st.selectbox("Eje Y (Numérico):", cols_num)
        x = st.selectbox("Eje X (Categoría):", [None] + cols_cat)
        st.plotly_chart(px.violin(df, y=y, x=x, box=True, points="all"), use_container_width=True)

    elif tipo == "Boxplot":
        y = st.selectbox("Eje Y (Numérico):", cols_num)
        st.plotly_chart(px.box(df, y=y, notched=True), use_container_width=True)

    elif tipo == "Barras con Error":
        x_col = st.selectbox("Categoría:", cols_cat)
        y_col = st.selectbox("Valor:", cols_num)
        df_err = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        st.plotly_chart(px.bar(df_err, x=x_col, y='mean', error_y='std', title="Media con Desviación Estándar"), use_container_width=True)

    elif tipo == "Correlación":
        st.plotly_chart(px.imshow(df[cols_num].corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
