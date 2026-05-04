import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Explorador de Datos Pro", layout="wide")

st.title("📊 Data Explorer & Correlation Lab")
st.markdown("""
Esta aplicación permite cargar un archivo CSV para explorar sus dimensiones y 
analizar la relación entre sus variables de forma dinámica.
""")

# --- Barra Lateral para Carga de Datos ---
st.sidebar.header("Configuración")
archivo_cargado = st.sidebar.file_uploader("Sube tu archivo CSV aquí", type=["csv"])

if archivo_cargado is not None:
    # Lectura de datos
    df = pd.DataFrame()
    try:
        df = pd.read_csv(archivo_cargado)
        st.success("✅ ¡Archivo cargado con éxito!")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

    # --- Sección 1: Vista Previa ---
    st.subheader("1. Vista previa de los datos (df.head)")
    st.dataframe(df.head(), use_container_width=True)
    
    # Métricas rápidas
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores Nulos", df.isna().sum().sum())

    # --- Sección 2: Análisis de Correlación ---
    st.divider()
    st.subheader("2. Diagrama de Correlación Personalizado")
    
    # Filtrar solo columnas numéricas para correlación
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(columnas_numericas) > 1:
        columnas_sel = st.multiselect(
            "Selecciona las columnas para analizar:",
            options=columnas_numericas,
            default=columnas_numericas[:min(5, len(columnas_numericas))]
        )
        
        if len(columnas_sel) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[columnas_sel].corr()
            
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f", 
                linewidths=0.5,
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.info("Selecciona al menos dos columnas para generar el mapa de calor.")
    else:
        st.warning("El archivo no tiene suficientes columnas numéricas para calcular correlaciones.")

else:
    st.info("Esperando archivo CSV... Por favor, súbelo desde la barra lateral.")
