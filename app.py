import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(page_title="Data Pipeline Sequencer", layout="wide")

# Inicializar el estado de la etapa si no existe
if 'etapa' not in st.session_state:
    st.session_state.etapa = 0

def avanzar_etapa(nueva_etapa):
    st.session_state.etapa = nueva_etapa

# Título dinámico según la etapa
st.title(f"🚀 Pipeline de Datos - Etapa {st.session_state.etapa}")

# ---------------------------------------------------------
# ETAPA 0: CARGA DE DATOS
# ---------------------------------------------------------
st.header("Etapa 0: Carga de Datos")
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    st.success("Archivo cargado correctamente.")
    st.subheader("df.head()")
    st.dataframe(df.head())
    
    if st.session_state.etapa == 0:
        if st.button("Confirmar carga y pasar a EDA"):
            avanzar_etapa(1)
            st.rerun()
else:
    st.info("Por favor, sube un archivo para comenzar.")
    st.stop() # Detiene la ejecución hasta que haya un archivo

# ---------------------------------------------------------
# ETAPA 1: EDA (Análisis Exploratorio)
# ---------------------------------------------------------
if st.session_state.etapa >= 1:
    st.divider()
    st.header("Etapa 1: Análisis Exploratorio (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dimensiones:**", df.shape)
        st.write("**Tipos de datos:**")
        st.write(df.dtypes)
    with col2:
        st.write("**Resumen Estadístico:**")
        st.write(df.describe())

    if st.session_state.etapa == 1:
        if st.button("Pasar a Limpieza"):
            avanzar_etapa(2)
            st.rerun()

# ---------------------------------------------------------
# ETAPA 2: LIMPIEZA
# ---------------------------------------------------------
if st.session_state.etapa >= 2:
    st.divider()
    st.header("Etapa 2: Limpieza de Datos")
    
    nulos = df.isnull().sum()
    st.write("Valores nulos por columna:")
    st.write(nulos[nulos > 0] if nulos.sum() > 0 else "No se detectaron valores nulos.")
    
    # Simulación de limpieza: Eliminar duplicados
    if st.checkbox("Eliminar filas duplicadas"):
        antes = len(df)
        df = df.drop_duplicates()
        st.write(f"Filas eliminadas: {antes - len(df)}")

    if st.session_state.etapa == 2:
        if st.button("Finalizar limpieza y ver Visualización"):
            avanzar_etapa(4) # Saltamos a la etapa 4 según tu solicitud
            st.rerun()

# ---------------------------------------------------------
# ETAPA 4: VISUALIZACIÓN
# ---------------------------------------------------------
if st.session_state.etapa >= 4:
    st.divider()
    st.header("Etapa 4: Visualización (Correlación)")
    
    # Filtrar solo columnas numéricas
    cols_num = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(cols_num) >= 2:
        st.subheader("Configura tu Diagrama de Correlación")
        columnas_sel = st.multiselect(
            "Selecciona las columnas:",
            options=cols_num,
            default=cols_num[:min(3, len(cols_num))]
        )
        
        if len(columnas_sel) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[columnas_sel].corr(), annot=True, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Selecciona al menos 2 columnas numéricas.")
    else:
        st.error("No hay suficientes datos numéricos para una correlación.")

    if st.button("Reiniciar Aplicación"):
        st.session_state.etapa = 0
        st.rerun()
